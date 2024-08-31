import torch

def get_fixed_npp_ex(
    ex_set,
    ex_per_phone = None,
    num_phones = 39):    
    r"""
    - num_training_frames:      Number of frames in the training set.
    - batch_size:               Training minibatch size.
    - ex_batch_size:            Size of the exemplar set for each minibatch.
    - ex_subset:                Subset of data for use as exemplars.
    - ex_per_phone:             Numer of exemplars per phone. If None, a non-stratified random
                                sample is used. If an int, that number is used per phone. If a list,
                                it should be of length num_phones.
    - ex_overlap                The number of exemplars to overlap between consecutive exemplar
                                sets. If int, it's a fixed number. If a list, it's the numbers per
                                phone.
    - avoid_minibatch_overlap:  Flag to prevent overlap between each minibatch and its exemplar
                                set.
    - IDX:                      The order that the training data is being used in. Only required
                                if the user wants to avoid overlap between the minibatch
                                and the exemplar set. If avoid_overlap=True and IDX is none, 
                                it is assumed there is no shuffling. 
    """

    ex_len = len(ex_set)
    shuffle = torch.randperm(ex_len)
    phones = ex_set.phones[shuffle]
    len_phones = torch.zeros(num_phones, dtype = torch.long)

    phones_idx = []
    for phoneID in range(num_phones):
        phones_idx.append(torch.arange(ex_len)[phones == phoneID])
        len_phones[phoneID] = len(phones_idx[phoneID])
    
    max_len = len_phones.max()

    for phoneID in range(num_phones):
        repeats, remainder = divmod(max_len.item(), len_phones[phoneID].item())
        phones_idx[phoneID] = phones_idx[phoneID].repeat(repeats)
        phones_idx[phoneID] = torch.cat((phones_idx[phoneID], phones_idx[phoneID][0:remainder]), dim = 0)
        
    phones_idx = torch.stack(phones_idx, dim = 1)
    ex_idx = shuffle[phones_idx]

    # if ex_per_phone > max_len:
    #     batch_ex_idx = torch.cat((ex_idx, ex_idx[0:ex_per_phone - max_len]))
    # else:
    batch_ex_idx = ex_idx[0:ex_per_phone]
    
    return batch_ex_idx.flatten()
        




# class get_unfixed_npp_ex():    
#     r"""
#     - num_training_frames:      Number of frames in the training set.
#     - batch_size:               Training minibatch size.
#     - ex_batch_size:            Size of the exemplar set for each minibatch.
#     - ex_subset:                Subset of data for use as exemplars.
#     - ex_per_phone:             Numer of exemplars per phone. If None, a non-stratified random
#                                 sample is used. If an int, that number is used per phone. If a list,
#                                 it should be of length num_phones.
#     - ex_overlap                The number of exemplars to overlap between consecutive exemplar
#                                 sets. If int, it's a fixed number. If a list, it's the numbers per
#                                 phone.
#     - avoid_minibatch_overlap:  Flag to prevent overlap between each minibatch and its exemplar
#                                 set.
#     - IDX:                      The order that the training data is being used in. Only required
#                                 if the user wants to avoid overlap between the minibatch
#                                 and the exemplar set. If avoid_overlap=True and IDX is none, 
#                                 it is assumed there is no shuffling. 
#     """
#     def __init__(
#     self,
#     ex_subset,
#     ex_per_phone = None,
#     fixed_ex = False,
#     num_phones = 39
#     ):
#         self.ex_subset = ex_subset
#         self.ex_per_phone = ex_per_phone
#         self.ex_len = len(ex_subset.indices)
#         self.num_phones = num_phones
#         self.shuffle = torch.randperm(self.ex_len)
#         self.reverseShuffle = torch.zeros_like(self.shuffle)
#         self.reverseShuffle[self.shuffle] = torch.arange(self.ex_len)
#         self.fixed_ex = fixed_ex

#         phones = ex_subset.dataset.phones[ex_subset.indices[self.shuffle]]
        
#         len_phones = torch.zeros(num_phones, dtype = torch.long)

#         phones_idx = []
#         for phoneID in range(num_phones):
#             phones_idx.append(torch.arange(self.ex_len)[phones == phoneID])
#             len_phones[phoneID] = len(phones_idx[phoneID])
        
#         max_len = len_phones.max()

#         for phoneID in range(num_phones):
#             repeats, remainder = divmod(max_len.item(), len_phones[phoneID].item())
#             phones_idx[phoneID] = phones_idx[phoneID].repeat(repeats)
#             phones_idx[phoneID] = torch.cat((phones_idx[phoneID], phones_idx[phoneID][0:remainder]), dim = 0)
            
#         phones_idx = torch.stack(phones_idx, dim = 1)
#         self.ex_idx = self.shuffle[phones_idx]
#         self.max_len = max_len

#         self.start = 0

#         self.current_ex_idx = None


#     def get_exIDX(self, ex_per_phone = None):

#         if self.fixed_ex and self.current_ex_idx is not None:
#             return self.current_ex_idx
#         else:

#             if ex_per_phone is None:
#                 ex_per_phone = self.ex_per_phone

#             if self.start + ex_per_phone > self.max_len:
#                 batch_ex_idx = torch.cat((self.ex_idx[self.start:], self.ex_idx[0:self.start + ex_per_phone - self.max_len]))
#                 self.start = self.start + ex_per_phone - self.max_len
#             else:
#                 batch_ex_idx = self.ex_idx[self.start:self.start + ex_per_phone]
#                 self.start += ex_per_phone
            
#             self.current_ex_idx = batch_ex_idx.flatten()
            
#             return self.current_ex_idx

