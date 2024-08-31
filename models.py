# import math
import torch
from torch import nn


class minerva2(nn.Module):
    def __init__(
        self, 
        p_factor = 1,
    ):
        super().__init__()

        self.p_factor = p_factor


    def forward(self, features, ex_features, ex_class_reps, p_factor = None):

        p_factor = p_factor if p_factor is not None else self.p_factor
        
        s = torch.matmul(
            nn.functional.normalize(features, dim = -1), 
            nn.functional.normalize(ex_features, dim = -1).transpose(dim0 = -2, dim1 = -1)
        )

        a = self.activation(s, p_factor)
        echo = torch.matmul(a, ex_class_reps)

        return echo, a

    
    def activation(self, s, p_factor = None):
        # Raise to a power while preserving sign

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))



class ffnn_wrapper_init(nn.Module):
    def __init__(
            self, 
            args
            ):
        super().__init__()

        self.args = args

        if args.pretrained_model is not None:
            self.load(args.pretrained_model)
        else:
            
            class_embed_dim = self.args.class_embed_dim if self.args.class_embed_dim is not None else self.args.num_classes

            self.layer0 = nn.Linear(args.input_dim, args.feat_embed_dim)
            self.do0 = nn.Dropout(p = args.do_class)
            self.activation0 = nn.ReLU()
            self.layer1 = nn.Linear(args.feat_embed_dim, class_embed_dim)
            self.do1 = nn.Dropout(p = args.do_class)
            self.activation1 = nn.ReLU()
            self.layer2 = nn.Linear(class_embed_dim, args.num_classes)
            self.do2 = nn.Dropout(p = args.do_class)


    def forward(self, features):

        logits = self.layer0(features)
        logits = self.do0(logits)
        logits = self.activation0(logits)

        logits = self.layer1(logits)
        logits = self.do1(logits)
        logits = self.activation1(logits)

        logits = self.layer2(logits)
        logits = self.do2(logits)

        output = {
            'logits': logits
        }

        return output
    

    def save(self, save_file):

        self.to('cpu')
        torch.save(
            {
                "args": self.args,
                "state_dict": self.state_dict()
            }, 
            save_file
        )
        self.to(self.args.device)


    def load(self, load_file):

        self.to('cpu')

        save_dict = torch.load(load_file)
        self.args = save_dict["args"]
        self.__init__(self.args)
        self.load_state_dict(save_dict["state_dict"])

        self.to(self.args.device)

    def count_parameters(self): 
        unlearned_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        learned_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return learned_params, unlearned_params



class minerva_transform(nn.Module):
    
    def __init__(
        self, 
        args,
        ex_features = None,
        ex_classes = None,
        ex_idx = None,
        pretrained_model = None
    ):
        super().__init__()

        if pretrained_model is not None:
            self.load(pretrained_model)
        else:

            self.args = args
            self.p_factor = args.p_factor
            self.feat_embed_dim = args.feat_embed_dim
            self.class_embed_dim = args.class_embed_dim
            self.train_class_reps = args.train_class_reps
            self.train_ex_classes = args.train_ex_classes
            self.num_classes = args.num_classes
            self.device = args.device

            if self.feat_embed_dim is not None:
                self.g = nn.Sequential(
                    nn.Linear(
                        in_features = args.input_dim,
                        out_features = self.feat_embed_dim
                    ),
                    nn.Dropout(p = args.do_feats)
                )

            self.do_class = nn.Dropout(p = args.do_class)
            
            if self.class_embed_dim is not None:
                class_reps = torch.rand(self.num_classes, self.class_embed_dim)
            else:
                class_reps = torch.eye(self.num_classes, dtype = torch.float)

            self.class_reps = nn.Parameter(class_reps, requires_grad = self.train_class_reps)

            if self.train_ex_classes:
                self.ex_class_reps = nn.Parameter(
                    self.class_reps[ex_classes]
                )

            self.f = minerva2(self.p_factor)

            self.ex_features = nn.Parameter(ex_features, requires_grad = False)
            if ex_features is not None:
                print("ex_features.size:", self.ex_features.size())
                
            # if ex_features is not None:
            #     self.register_buffer('ex_features', ex_features)
            #     print("ex_features.size:", self.ex_features.size())
            # else:
            #     self.register_buffer('ex_features', None)

            if ex_classes is not None:
                self.register_buffer('ex_classes', ex_classes)
                print("ex_classes.size:", self.ex_classes.size())
            else:
                self.register_buffer('ex_classes', None)
            
            if ex_idx is not None:
                self.register_buffer('ex_idx', ex_idx)
                print("ex_idx.size:", self.ex_idx.size())
            else:
                self.register_buffer('ex_idx', None)
            

    def forward(self, features, p_factor = None):

        p_factor = p_factor if p_factor is not None else self.p_factor
        ex_features = self.ex_features

        if self.feat_embed_dim is not None:
            features = self.g(features)
            ex_features = self.g(ex_features)

        if self.train_ex_classes:
            ex_class_reps = self.ex_class_reps
        else:
            ex_class_reps = self.class_reps[self.ex_classes]

        ex_class_reps = self.do_class(ex_class_reps)

        echo, activations = self.f(
            features = features,
            ex_features = ex_features,
            ex_class_reps = ex_class_reps,
            p_factor = p_factor
        )
        
        class_reps = nn.functional.normalize(self.class_reps, dim = -1)

        logits = torch.matmul(
            nn.functional.normalize(echo, dim = -1), 
            class_reps.t()
        )

        if self.class_embed_dim is None:
            non_diag_loss = nn.functional.normalize(self.class_reps) - torch.eye(self.num_classes, device = self.device, dtype = torch.float)
            non_diag_loss = non_diag_loss.norm()
        else:
            non_diag_loss = torch.tensor(0, dtype = torch.float, device = self.device)

        output = {
            'echo': echo,
            'activations': activations,
            'logits': logits,
            'non_diag_loss': non_diag_loss
        }

        return output


    def save(self, save_file):

        self.to('cpu')
        torch.save(
            {
                "args": self.args,
                "state_dict": self.state_dict()
            }, 
            save_file
        )
        self.to(self.args.device)


    def load(self, load_file):

        self.to('cpu')
        save_dict = torch.load(load_file)
        state_dict = save_dict["state_dict"]
        args = save_dict["args"]
        self.__init__(
            args,
            ex_features = state_dict["ex_features"],
            ex_classes = state_dict["ex_classes"],
            ex_idx = state_dict["ex_idx"]
        )
        self.load_state_dict(state_dict)
        self.to(self.args.device)


    def count_parameters(self): 
        unlearned_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        learned_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return learned_params, unlearned_params




