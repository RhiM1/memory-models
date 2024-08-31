import torch
from torch import nn
import argparse
from datasets import get_timit_datasets
from models import minerva_transform, ffnn_wrapper_init
import time
from exemplar_select import get_fixed_npp_ex
import wandb


def test_loop(
    testset, 
    model, 
    lossFn,
    args,
    confusion = False
    ):

    with torch.no_grad():
        model.eval()

        num_frames = len(testset)
        num_batches, remainder = divmod(num_frames, args.batch_size)
        if remainder > 0:
            num_batches += 1

        allFeatures = testset.features.to(args.device)
        allPhones = testset.phones.to(args.device)

        if confusion:
            confMat = torch.zeros(args.num_classes, args.num_classes)
        else:
            confMat = None

        testLoss, correct = 0.0, 0

        for batchID in range(num_batches):
            IDX = torch.arange(num_frames)[batchID * args.batch_size:(batchID + 1) * args.batch_size]
            features = allFeatures[IDX]
            phones = allPhones[IDX]
            output = model(features)
            preds = output['logits']

            if confusion:
                for truePhone, predPhone in zip(phones, preds.argmax(1)):
                    confMat[truePhone.long(), predPhone.long()] += 1

            correct += (preds.argmax(1) == phones).type(torch.float).sum().item() 
            testLoss += lossFn(preds, phones).item()
    
    correct /= num_frames
    testLoss /= num_batches

    return correct, testLoss, confMat


def train_loop(
    args,
    trainset, 
    model, 
    lossFn,
    optimizer
    ):
    r"""
    Mode options:
    - dataloader:       Loads the training data.

    - exemplars:        - If the mode is "nPP", a dataloader using a 
                          stratifiedExemplarSampler as its sampler.
                        - If mode is "noEx", unused - can be None.

    - model             The neural network.

    - Mode:
        randSample:     Select M exemplars in total from the training data that is
                        not being used for teh current minibatch.
        nPP:            Select N exemplars per phone from the training data.
                        Currently, these exemplars may overlap with the current 
                        minibatch, if both exemplars and inputs are from the same
                        set (which is typical during training).
        noExemplars:    An empty exemplar set is used, and exemplar-based
                        parameters are not trained. 
        cheat:          Use the current minibatch for exemplars, which will 
                        inevitably include the true answer.

    """

    model.train()

    num_frames = len(trainset)
    num_batches, remainder = divmod(num_frames, args.batch_size)
    if remainder > 0:
        num_batches += 1

    allFeatures = trainset.features.to(args.device)
    allPhones = trainset.phones.to(args.device)

    trainLoss, correct = 0.0, 0

    allIDX = torch.randperm(num_frames)

    for batchID in range(num_batches):
        IDX = allIDX[batchID * args.batch_size:(batchID + 1) * args.batch_size]

        phones = allPhones[IDX]
        features = allFeatures[IDX]

        phones = phones
        features = features
        output = model(features)
        preds = output['logits']

        loss = lossFn(preds, phones)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (preds.argmax(dim = 1) == phones).type(torch.float).sum().item() 
        trainLoss += loss.item()

    correct /= num_frames
    trainLoss /= num_batches

    return correct, trainLoss


def train(
    args,
    trainset, 
    devset,
    testset, 
    epochs,
    model, 
    lossFn,
    optimizer,
    epochs_start = 0):

    width = 14
    learned_params, unlearned_params = model.count_parameters()
    
    devCorrect, devLoss, _ = test_loop(
        testset = devset,
        model = model,
        lossFn = lossFn,
        args = args
    )    

    testCorrect, testLoss, _ = test_loop(
        testset = testset,
        model = model,
        lossFn = lossFn,
        args = args
    )

    model.save(args.model_file)
    torch.save(
        {
            'epoch': epochs_start,
            'trainAcc': None,
            'trainLoss': None,
            'devAcc': devCorrect,
            'devLoss': devLoss,
            "optimizer": optimizer.state_dict()
        },
        args.opt_file
    )

    current_time = time.strftime("%H-%M-%S %d-%b-%Y", time.localtime())
    print(f"Start time: {current_time}\n")
    print(f"Epoch {'train loss':>{width}} {'train correct':>{width}} {'dev loss':>{width}} {'dev correct':>{width}} {'test loss':>{width}} {'test correct':>{width}}")
    print(f"{epochs_start:>5} {'':>{width}} {'':>{width}} {devLoss:>{width}.4f} {devCorrect:>{width}.4f} {testLoss:>{width}.4f} {testCorrect:>{width}.4f}")

    with open(args.log_file, 'a') as f:
        if not args.skip_wandb:
            f.write(f"\nWandb project: {args.wandb_project}\nInitial wandb ID: {wandb.run.id}\n")
        else:
            f.write(f"\nWandb project: {args.wandb_project}\nInitial wandb ID: {None}\n")
        strWrite = f"\nModel: {args.model_name}\nLearned parameters: {learned_params}, Unlearned parameters: {unlearned_params}\n\n"
        strWrite += f"\nepoch \t trainLoss \t trainCorrect"
        strWrite += f" \t devLoss \t devCorrect"
        strWrite += f" \t testLoss \t testCorrect"
        strWrite += "\n"
        f.write(strWrite)
        strWrite = f"{epochs_start}    "
        strWrite += f" \t           \t              "
        strWrite += f" \t {devLoss:>0.4}      \t {devCorrect:>0.4}        "
        strWrite += f" \t {testLoss:>0.4}      \t {testCorrect:>0.4}        "
        strWrite += "\n"
        f.write(strWrite)
    
    if not args.skip_wandb:
        wandb.log({
            'dev_loss': devLoss, 
            'dev_acc': devCorrect, 
            'test_loss': testLoss, 
            'test_acc': testCorrect, 
            'epoch': epochs_start
        })

    bestDevCorrect = devCorrect

    for epoch in range(epochs_start + 1, epochs + 1):
        trainCorrect, trainLoss = train_loop(
            args = args,
            trainset = trainset,
            model = model,
            lossFn = lossFn,
            optimizer = optimizer
        )

        devCorrect, devLoss, _ = test_loop(
            testset = devset,
            model = model,
            lossFn = lossFn,
            args = args
        )

        testCorrect, testLoss, _ = test_loop(
            testset = testset,
            model = model,
            lossFn = lossFn,
            args = args
        )
        
        if not args.skip_wandb:
            wandb.log({
                'train_loss': trainLoss, 
                'train_acc': trainCorrect, 
                'dev_loss': devLoss, 
                'dev_acc': devCorrect, 
                'test_loss': testLoss, 
                'test_acc': testCorrect, 
                'epoch': epoch
            })
        
        strWrite = f"Epoch {epoch}"
        strWrite += f" \t {trainLoss:>0.4f} \t {trainCorrect:>0.4f}"
        strWrite += f" \t {devLoss:>0.4} \t {devCorrect:>0.4}"
        strWrite += f" \t {testLoss:>0.4} \t {testCorrect:>0.4}"
        
        with open(args.log_file, 'a') as f:
            f.write(strWrite + "\n")

        strWrite = f"{epoch:>5} {trainLoss:>{width}.4f} {trainCorrect:>{width}.4f} {devLoss:>{width}.4f} {devCorrect:>{width}.4f} {testLoss:>{width}.4f} {testCorrect:>{width}.4f}"

        if  devCorrect > bestDevCorrect:
            strWrite += f"  Best so far - Saving model..."

            model.save(args.model_file)
            torch.save(
                {
                    'epoch': epoch,
                    'trainAcc': trainCorrect,
                    'trainLoss': trainLoss,
                    'devAcc': devCorrect,
                    'devLoss': devLoss,
                    "optimizer": optimizer.state_dict()
                },
                args.opt_file
            )

            bestDevCorrect = devCorrect
        
        print(strWrite)
        

def start_wandb(args):
        
    config = vars(args)
    
    run = wandb.init(
        project=args.wandb_project, 
        reinit = True, 
        name = args.model_name,
        config = config,
        tags = [
            f"e{args.exp_id}",
            f"r{args.run_id}",
            f"{args.input_feats}{args.feats_layer}",
            f"{args.model}",
            f"bs{args.batch_size}", 
            f"lr{args.lr}", 
            f"wd{args.wd}",
            f"dof{args.do_feats}",
            f"doc{args.do_class}",
            f"fd{args.feat_embed_dim}",
            f"cd{args.class_embed_dim}",
            # f"ln{int(args.use_layer_norm)}",
            f"s{args.seed}",
            # f"dat_prop{args.prop_train_data}",
            f"rs{int(args.random_sample)}",
        ]
    )
    
    # if args.model == 'ffnn_init':
    #     run.tags = run.tags + (
    #         # f"mod_init_{args.model_init}",
    #         # f"act0_{args.act0}",
    #         # f"act1_{args.act1}",
    # )
    if args.model == 'minerva':
        run.tags = run.tags + (
            f"epc{args.ex_per_class}",
            f"p{args.p_factor}",
            # f"sm{int(args.use_sm)}",
            f"tcr{int(args.train_class_reps)}",
            f"tec{int(args.train_ex_classes)}",
            f"lrcr{args.lr_cr}",
            f"lrex{args.lr_ex}",
            f"wdcr{args.wd_cr}",
            f"wdex{args.wd_ex}",
        )

    print(f'\nLogging with Wandb id: {wandb.run.id}\nName: {args.exp_id}_{args.model_name}_{args.input_feats}')

    return run



def main(args):

    useWANDB = not args.skip_wandb             

    start_time = time.strftime("%H-%M-%S %d-%b-%Y", time.localtime())

    with open(args.log_file, 'a') as f:
        f.write(f"\nStart time: {start_time}\n")

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)

    print("Loading data...")
    trainingData, devData, testData = get_timit_datasets(args)
    print("Data loaded")

    ex_idx = get_fixed_npp_ex(
        trainingData,
        ex_per_phone = args.ex_per_class,
        num_phones = args.num_classes
    )
    ex_features = trainingData.features[ex_idx]
    ex_phones = trainingData.phones[ex_idx]

    torch.manual_seed(args.seed * 2)

    if args.model == "ffnn_init":
        model = ffnn_wrapper_init(args)
    elif args.model == "minerva":
        model = minerva_transform(
            args = args,
            ex_features = ex_features,
            ex_classes = ex_phones,
            ex_idx = ex_idx
        )

    print(args.model_name)
    print(model)

    if args.pretrained_model is not None:
        model.load(args.pretrained_model)

    learned_params, unlearned_params = model.count_parameters()
    print(f"Model: {args.model_name}\nLearned parameters: {learned_params}, Unlearned parameters: {unlearned_params}\n")

    model.to(args.device)

    lossFn = nn.CrossEntropyLoss()

    if not args.skip_train:

        optParams = []

        for name, param in model.named_parameters():
            if name == "class_reps" or name == "class_trans.weight" or name == "class_trans.bias":
                optParams.append(
                    {'params': param, 'weight_decay': args.wd_cr, 'lr': args.lr_cr}
                )
            elif name == "ex_class_reps":
                optParams.append(
                    {'params': param, 'weight_decay': args.wd_ex, 'lr': args.lr_ex}
                )
            else:
                optParams.append(
                    {'params': param, 'weight_decay': args.wd, 'lr': args.lr}
                )
        optimizer = torch.optim.Adam(optParams)
    
        if useWANDB:
            run = start_wandb(args)

        train(
            args,
            trainingData,
            devData,
            testData,
            args.epochs,
            model,
            lossFn,
            optimizer
        )
        
        model.load(args.model_file)
        opt = torch.load(args.opt_file)
        epoch = opt["epoch"]
        trainCorrect = opt["trainAcc"]
        trainLoss = opt["trainLoss"]
        devCorrect = opt["devAcc"]
        devLoss = opt["devLoss"]

    devCorrect, devLoss, _ = test_loop(
        testset = devData,
        model = model,
        lossFn = lossFn,
        args = args,
        confusion = False
    )   
        
    testCorrect, testLoss, confMat = test_loop(
        testset = testData,
        model = model,
        lossFn = lossFn,
        args = args,
        confusion = True
    )
    if args.skip_train:

        if useWANDB:
            run = start_wandb(args)
        
        width = 14
        
        print(f"Epoch {'train loss':>{width}} {'train correct':>{width}} {'dev loss':>{width}} {'dev correct':>{width}} {'test loss':>{width}} {'test correct':>{width}}")
        print(f"{0:>5} {'':>{width}} {'':>{width}} {devLoss:>{width}.4f} {devCorrect:>{width}.4f} {testLoss:>{width}.4f} {testCorrect:>{width}.4f}")

    
    if useWANDB:
        wandb.log({
            'best_dev_acc': devCorrect, 
            'best_test_acc': testCorrect, 
            'best_epoch': 0 if args.skip_train else epoch
        })

    finishTime = time.strftime("%H-%M-%S %d-%b-%Y", time.localtime())
    
    with open(args.log_file, 'a') as f:
        epoch = 0 if args.skip_train else epoch
        f.write(f"\nBest epoch: {epoch}\n")
        strWrite = ""
        if args.skip_train or trainLoss is None or trainCorrect is None:
            strWrite += f" \t None \t None"
        else:
            strWrite += f" \t {trainLoss:>0.4f} \t {trainCorrect:>0.3f}"
        strWrite += f" \t {devLoss:>0.4} \t {devCorrect:>0.3}"
        strWrite += f" \t {testLoss:>0.4} \t {testCorrect:>0.3}"
        strWrite += "\n"
        f.write(strWrite)

        confRows = []
        for row in confMat:
            confRows.append("\t".join(str(val.item()) for val in row))
        f.write("\nConfusion matrix FF\n")
        f.write("\n".join(confRows))
        f.write("\n")

        f.write(f"\nFinish time: {finishTime}\n")
    with open(args.summ_file, 'a') as f:
        f.write(f"{args.model_name} \t {epoch} \t " + strWrite)

    if useWANDB:
        run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--exp_id", help="id for experiment project", default = "000"
    )
    parser.add_argument(
        "--run_id", help="id for individual experiment", default = "000"
    )
    parser.add_argument(
        "--exp_folder", help = "experiment folder name" , default = "exp"
    )
    parser.add_argument(
        "--summ_file", help="path to write summary results to" , default="summary.log"
    )
    parser.add_argument(
        "--wandb_project", help = "W and B project name" , default = "mm"
    )
    parser.add_argument(
        "--skip_wandb", help="skip logging via WandB", default=False, action='store_true'
    )
    parser.add_argument(
        "--seed", help="random seed for repeatability", default=42, type=int
    )

    # Data details
    parser.add_argument(
        "--input_feats", help="feats to be used: MelSpec, wav2vec, hubert11, " , default="hubert",
    )
    parser.add_argument(
        "--feats_layer", help="where appropriate, the feature extractor layer to use" , default=11, type = int,
    )
    parser.add_argument(
        "--num_classes", help="number of classes" , default=39, type = int,
    )

    # Model details
    parser.add_argument(
        "--model", help="phone classification model: ffnn, minerva2, minerva3", default="ffnn"
    )
    parser.add_argument(
        "--pretrained_model", help="location of pretrained model file", default=None
    )
    parser.add_argument(
        "--feat_embed_dim", help="exemplar model feat transformation dimension" , default=None, type=int
    )
    parser.add_argument(
        "--class_embed_dim", help="dimension of class representation", default=None, type=int
    )


    # Exemplar model details
    parser.add_argument(
        "--p_factor", help="exemplar model p_factor" , default=1, type=float
    )
    parser.add_argument(
        "--ex_per_class", help="number of exemplars per class", default=384, type=int
    )
    parser.add_argument(
        "--train_class_reps", help="train the 'true' class representations", default=False, action='store_true'
    )
    parser.add_argument(
        "--train_ex_classes", help="train each exemplar class", default=False, action='store_true'
    )
    parser.add_argument(
        "--train_ex_feats", help="train each exemplar feature", default=False, action='store_true'
    )
    parser.add_argument(
        "--random_sample", help="use a random sample, rather than N per class", default=False, action='store_true'
    )

    # Training hyperparameters
    parser.add_argument(
        "--skip_train", help="skip training", default=False, action='store_true'
    )
    parser.add_argument(
        "--batch_size", help="batch size" , default=1024, type=int
    )
    parser.add_argument(
        "--epochs", help="number of epochs", default=150, type=int
    )
    parser.add_argument(
        "--lr", help="learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--lr_cr", help="learning rate", default=None, type=float
    )
    parser.add_argument(
        "--lr_ex", help="learning rate", default=None, type=float
    )
    parser.add_argument(
        "--wd", help="weight decay", default=0, type=float
    )
    parser.add_argument(
        "--wd_cr", help="weight decay for trained exemplar representations", default=None, type=float
    )

    parser.add_argument(
        "--wd_ex", help="weight decay for trained exemplar representations", default=0, type=float
    )
    parser.add_argument(
        "--do_feats", help="dropout for feats transform", default=0, type=float
    )
    parser.add_argument(
        "--do_class", help="dropout for classifier", default=0, type=float
    )

    args = parser.parse_args()

    if args.pretrained_model is not None:
        args.pretrained_model = f"{args.exp_folder}/models/{args.pretrained_model}"

    if args.model == 'ffnn_init':
        args.ex_per_class = (args.feat_embed_dim // args.num_classes) + 1
        args.num_ex = args.feat_embed_dim
    else:
        args.num_ex = args.ex_per_class * args.num_classes


    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.input_feats == "MelSpec":
        args.input_dim = 96
        args.feats_layer = ""
    elif args.input_feats  == "wav2vec" or args.input_feats == "hubert":
        args.input_dim = 768

    if args.lr_cr is None:
        args.lr_cr = args.lr
    
    if args.wd_cr is None:
        args.wd_cr = args.wd
    
    args.data_folder = "data/" + args.input_feats + "/"

    args.model_name = args.exp_id + \
        f"_{args.run_id}" + \
        f"_{args.model}" + \
        f"_{args.seed}"
    

    args.summ_file = args.exp_folder + "/" + args.summ_file
    args.model_file = args.exp_folder + "/models/" + args.model_name + ".mod"
    args.opt_file = args.exp_folder + "/models/" + args.model_name + ".opt"
    args.log_file = args.exp_folder + "/logs/" + args.model_name + ".txt"

    main(args)