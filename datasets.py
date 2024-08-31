import torchaudio
import torch
from torch.utils.data import Dataset
import time
from functions import get_phone_dicts
import os

DATAROOT = "/home/acp20rm/exp/data/TIMIT"
# DATAROOT = "/store/store1/data/TIMIT"


spk_test = [
    "MDAB0", "MWBT0", "FELC0", "MTAS1", "MWEW0", "FPAS0", 
    "MJMP0", "MLNT0", "FPKT0", "MLLL0", "MTLS0", "FJLM0", 
    "MBPM0", "MKLT0", "FNLP0", "MCMJ0", "MJDH0", "FMGD0", 
    "MGRT0", "MNJM0", "FDHC0", "MJLN0", "MPAM0", "FMLD0"
]

spk_dev = [
    "fadg0", "faks0", "fcal1", "fcmh0", "fdac1",
    "fdms0", "fdrw0", "fedw0", "fgjd0", "fjem0",
    "fjmg0", "fjsj0", "fkms0", "fmah0", "fmml0",
    "fnmr0", "frew0", "fsem0", "majc0", "mbdg0",
    "mbns0", "mbwm0", "mcsh0", "mdlf0", "mdls0",
    "mdvc0", "mers0", "mgjf0", "mglb0", "mgwt0",
    "mjar0", "mjfc0", "mjsw0", "mmdb1", "mmdm2",
    "mmjr0", "mmwh0", "mpdf0", "mrcs0", "mreb0",
    "mrjm4", "mrjr0", "mroa0", "mrtk0", "mrws1",
    "mtaa0", "mtdt0", "mteb0", "mthc0", "mwjg0",
]


class custom_timit_dataset(Dataset):
    def __init__(
        self, 
        features, 
        phones, 
        spk_id,
        reg_id,
        gen,
        utt_id,
        id_to_phone,
        phone_to_id,
        spk_to_id,
        id_to_spk,
        ctc_phones = None,
        sequence_based = False
    ):

        self.features = features
        self.phones = phones
        self.spk_id = spk_id
        self.reg_id = reg_id
        self.gen = gen
        self.utt_id = utt_id
        self.id_to_phone = id_to_phone
        self.phone_to_id = phone_to_id
        self.spk_to_id = spk_to_id
        self.id_to_spk = id_to_spk
        self.sequence_based = sequence_based
        self.ctc_phones = ctc_phones

     
    def __len__(self):
        return len(self.phones)
    

    def __getitem__(self, idx):
        if not self.sequence_based:
            return self.f_spk_id[idx], self.f_gen[idx], self.f_utt_id[idx], self.f_reg_id[idx], self.f_features[idx], self.f_phones[idx]
        if self.ctc_phones is None:
            return self.spk_id[idx], self.gen[idx], self.utt_id[idx], self.reg_id[idx], self.features[idx], self.phones[idx]
        else:
            return self.spk_id[idx], self.gen[idx], self.utt_id[idx], self.reg_id[idx], self.features[idx], self.phones[idx], self.ctc_phones[idx]
    

def get_timit_datasets(
    args,
    keepQ = False
    ):

    input_feats = args.input_feats
    device = args.device
    layer = args.feats_layer
    num_phones = args.num_classes
    
    if hasattr(args, 'phone_context'):
        phone_context = args.phone_context
    else:
        phone_context = 0
    
    if hasattr(args, 'sequence_based'):
        sequence_based = args.sequence_based
    else:
        sequence_based = False

    start_time = time.strftime("%H-%M-%S %d-%b-%Y", time.localtime())
    print(start_time)

    if input_feats == 'mfcc':
        stride = 160
        feat_model = torchaudio.transforms.MFCC(
            sample_rate = 16000,
            n_mfcc = 13,
            melkwargs={"n_fft": 400, "hop_length": stride, "n_mels": 23, "center": False}
        )
    if input_feats == "MelSpec":
        stride = 320
        feat_model = torchaudio.transforms.MelSpectrogram(
            n_fft = 512,
            n_mels = 32,
            hop_length = stride
        )
    elif input_feats == 'wav2vec':
        stride = 320
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        feat_model = bundle.get_model().to(device)
    elif input_feats == 'hubert':
        stride = 320
        bundle = torchaudio.pipelines.HUBERT_BASE
        feat_model = bundle.get_model().to(device)

    spk_to_id, id_to_spk = {}, {}

    utt_id = []
    split = []
    spk_id = []
    reg_id = []
    gen = []
    feats = []
    phones = []

    i = 0
    speaker_id = 0

    num_test = 0
    num_dev = 0
    num_train = 0
    
    phone_to_id, id_to_phone = get_phone_dicts("utils/phones.map", num_phones = num_phones)

    for root, dirs, files in os.walk(DATAROOT):
        for file in files:
            if file.endswith(".WAV") and not file.endswith("SA1.WAV") and not file.endswith("SA2.WAV"):
                root_list = root.split("/")
                aSplit = root_list[-3]
                speaker = root_list[-1]
                # Use all TRAIN data, but restrict dev and test sets to their specific speakers
                if aSplit == "TRAIN" or speaker.lower() in spk_dev or speaker in spk_test:
                    gender = 0 if speaker[0] == "F" else 1
                    region = root_list[-2][-1]
                    sentence = file.split(".")[0]
                    wav, _ = torchaudio.load(f"{root}/{file}", normalize=True)
                    if speaker not in spk_to_id:
                        spk_to_id[speaker] = speaker_id
                        id_to_spk[speaker_id] = speaker
                        speaker_id += 1
                    utt_id.append(i)
                    if aSplit == "TEST" and speaker in spk_test:
                        split.append(2)
                        num_test += 1
                    elif aSplit == "TEST" and speaker.lower() in spk_dev:
                        split.append(1)
                        num_dev += 1
                    else:
                        split.append(0)
                        num_train += 1
                    if input_feats == 'mfcc':
                        feat = feat_model(wav)
                        delta = torchaudio.functional.compute_deltas(feat)
                        delta2 = torchaudio.functional.compute_deltas(delta)
                        feat = torch.cat((feat, delta, delta2), dim = 1)
                        feat = feat.squeeze()
                        feat = feat.t()
                    elif input_feats == "MelSpec":
                        feat = feat_model(wav)
                        feat = torch.log(feat)
                        delta = torchaudio.functional.compute_deltas(feat)
                        delta2 = torchaudio.functional.compute_deltas(delta)
                        feat = torch.cat((feat, delta, delta2), dim = 1)
                        feat = feat.squeeze()
                        feat = feat.t()
                    elif input_feats == 'hubert' or input_feats == 'wav2vec':
                        wav = wav.to(device)
                        feat, _ = feat_model.extract_features(wav)
                        feat = feat[layer].detach().cpu()[0]
                    spk_id.append(spk_to_id[speaker])
                    reg_id.append(int(region))
                    gen.append(gender)
                    feats.append(feat)

                    # Get phone labels
                    with open(f"{root}/{sentence}.PHN") as f:
                        phone_data = f.read().split("\n")
                    remainder = 0
                    starts = []
                    stops = []
                    summ_labels = []
                    # Some label files are missing the first row, so add it in
                    if phone_data[0][0] != '0':
                        starts.append(0)
                        stops.append(int(phone_data[0].split()[0]))
                        summ_labels.append(0)
                    for row in phone_data:
                        if row != "":
                            start, stop, phone = row.split()
                            starts.append(int(start))
                            stops.append(int(stop))
                            phone = phone_to_id[phone]
                            summ_labels.append(phone)
                    
                    labels = []
                    summ_labels = torch.tensor(summ_labels, dtype = torch.long)
                    context_phones = torch.zeros(2 * phone_context + 1, dtype = torch.long)
                    context_phones[phone_context + 1:] = summ_labels[0:phone_context]
                    for i in range(len(summ_labels)):
                        start = starts[i]
                        stop = stops[i]
                        num_repeats, remainder = divmod(remainder + stop - start, stride)
                        if i + phone_context < len(summ_labels):
                            context_phones[0:-1] = context_phones.clone()[1:]
                            context_phones[-1] = summ_labels[i + phone_context]
                            labels.append(context_phones.repeat(num_repeats, 1))
                        else:
                            context_phones[0:-1] = context_phones.clone()[1:]
                            context_phones[-1] = 0
                            labels.append(context_phones.repeat(num_repeats, 1))
                
                    labels = torch.cat(labels, dim = 0)

                    # Pad / trim the labels to be the same length as the 
                    if len(labels) > len(feats[-1]):
                        if len(labels) > len(feats[-1]) + 2:
                            print(f"phone labels too long! {len(labels), len(feats[-1])}")
                        labels = labels[0:len(feats[-1])]
                    elif len(labels) < len(feats[-1]):
                        xtend = labels[-1]
                        if labels[-1, phone_context] != 0:
                            xtend[0:-1] = xtend[1:].clone()
                            xtend[-1] = 0
                        xtend = xtend.repeat(len(feats[-1]) - len(labels), 1)
                        labels = torch.cat([labels, xtend], dim = 0)

                    phones.append(labels)
                
                    i += 1

    utt_ids = []
    splits = []
    spk_ids = []
    reg_ids = []
    gens = []
    features = []

    for i, feat in enumerate(feats):
        features.append(feat)
        if not sequence_based:
            utt_ids = utt_ids + [utt_id[i]] * len(feat)
            splits.extend([split[i]] * len(feat))
            spk_ids.extend([spk_id[i]] * len(feat))
            reg_ids.extend([reg_id[i]] * len(feat))
            gens.extend([gen[i]] * len(feat))
        elif not keepQ:
            features[i] = features[i][phones[i].squeeze() != 39]
            phones[i] = phones[i][phones[i].squeeze() != 39]

    if sequence_based:
        utt_id = torch.tensor(utt_id)
        split = torch.tensor(split)
        spk_id = torch.tensor(spk_id)
        reg_id = torch.tensor(reg_id)
        gen = torch.tensor(gen)
        feats = features
    else:
        feats = torch.cat(features)
        phones = torch.cat(phones)
        utt_id = torch.tensor(utt_ids)
        split = torch.tensor(splits)
        spk_id = torch.tensor(spk_ids)
        reg_id = torch.tensor(reg_ids)
        gen = torch.tensor(gens)

        if not keepQ:
            utt_id = utt_id[phones[:, phone_context] != 39]
            split = split[phones[:, phone_context] != 39]
            spk_id = spk_id[phones[:, phone_context] != 39]
            reg_id = reg_id[phones[:, phone_context] != 39]
            gen = gen[phones[:, phone_context] != 39]
            feats = feats[phones[:, phone_context] != 39]
            phones = phones[phones[:, phone_context] != 39]
    
    finishTime = time.strftime("%H-%M-%S %d-%b-%Y", time.localtime())
    print(finishTime)

    train_feats = []
    train_phones = []
    dev_feats = []
    dev_phones = []
    test_feats = []
    test_phones = []
    if sequence_based:
        for i, feat in enumerate(features):
            if split[i] == 0:
                train_feats.append(feat)
                train_phones.append(phones[i])
            elif split[i] == 1:
                dev_feats.append(feat)
                dev_phones.append(phones[i])
            elif split[i] == 2:
                test_feats.append(feat)
                test_phones.append(phones[i])
    else:
        train_feats = feats[split == 0]
        dev_feats = feats[split == 1]
        test_feats = feats[split == 2]
        train_phones = phones[split == 0].squeeze()
        dev_phones = phones[split == 1].squeeze()
        test_phones = phones[split == 2].squeeze()

    train_data = custom_timit_dataset(
        train_feats,
        train_phones,
        spk_id[split == 0],
        reg_id[split == 0],
        gen[split == 0],
        utt_id[split == 0],
        id_to_phone,
        phone_to_id,
        spk_to_id,
        id_to_spk
    )

    dev_data = custom_timit_dataset(
        dev_feats,
        dev_phones,
        spk_id[split == 1],
        reg_id[split == 1],
        gen[split == 1],
        utt_id[split == 1],
        id_to_phone,
        phone_to_id,
        spk_to_id,
        id_to_spk
    )
    
    test_data = custom_timit_dataset(
        test_feats,
        test_phones,
        spk_id[split == 2],
        reg_id[split == 2],
        gen[split == 2],
        utt_id[split == 2],
        id_to_phone,
        phone_to_id,
        spk_to_id,
        id_to_spk
    )
    print(f"train size: {len(train_data)}, dev size: {len(dev_data)}, test size: {len(test_data)}")


    return train_data, dev_data, test_data

