import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from einops import rearrange

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):               
        x = self.conv(x)
        x = self.bn(x)
        return x
        
class ALTGCN(nn.Module):
    def __init__(self, in_channels, out_channels, Frames, kernel_size, stride=1):
        super(ALTGCN, self).__init__()
        pad = (kernel_size - 1) // 2
        self.Frames = Frames
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.DecoupleT = nn.Parameter(torch.ones([1,Frames,Frames], dtype=torch.float32, requires_grad=True).repeat(8,1,1), requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels)        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.avg_pool_t = nn.AdaptiveAvgPool2d(1)
        self.max_pool_t = nn.AdaptiveMaxPool2d(1)
        self.conv2_t = nn.Conv2d(2, 1, kernel_size=1, padding=0) 
        self.sigmoid = nn.Sigmoid()
        
        if in_channels != out_channels:
            self.down = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.down = lambda x: x        

    def forward(self, x):
    

        m = self.conv(x)
        N,C,T,V = m.size()
        
        DT = self.DecoupleT #
        norm_learn_T = DT.repeat(self.out_channels//8,1,1) # C Frames Frames        
        m = torch.einsum('nctv,ctq->ncqv', (m, norm_learn_T))        

        # T-ACM        
        m = m.permute(0,2,1,3).contiguous() # N T C V
        q = torch.cat([self.avg_pool_t(m),self.max_pool_t(m)],dim=2)
        q = q.permute(0,2,1,3).contiguous()
        q = self.conv2_t(q)
        q = q.permute(0,2,1,3).contiguous()        
        q = self.sigmoid(q)
        q = m + m  *  q.expand_as(m)        
        q = q.permute(0,2,1,3).contiguous()

        q = self.bn(q)
        q += self.down(x)         
        return self.relu(q)
        
       


class MSTFCN(nn.Module):
    def __init__(self, in_channels, out_channels, Frames, kernel_size=3, stride=1, dilations=[1,2,3,4], residual=True, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 3) == 0, '# out channels should be multiples of # branches'
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 3
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(in_channels,branch_channels,kernel_size=1,padding=0),nn.BatchNorm2d(branch_channels),nn.ReLU(inplace=True),
            TemporalConv(branch_channels,branch_channels,kernel_size=ks,stride=stride,dilation=dilation),)
            for ks, dilation in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            ALTGCN(in_channels, branch_channels, Frames, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))        
       
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)


    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class ALSGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups=8, coff_embedding=4, num_subset=3,t_stride=1,t_padding=0,t_dilation=1,bias=True):
        super(ALSGCN, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.groups=groups
        self.out_channels=out_channels
        self.alpha = nn.Parameter(torch.zeros(1))
        self.tan = nn.Tanh()
      
        self.num_subset = 3
        self.GroupA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32),[3,1,17,17]), dtype=torch.float32, requires_grad=True).repeat(1,groups,1,1), requires_grad=True)
        self.GroupMA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32),[3,1,17,17]), dtype=torch.float32, requires_grad=True).repeat(1,groups,1,1), requires_grad=True)
        self.A = Variable(torch.from_numpy(np.reshape(A.astype(np.float32),[3,1,17,17]).repeat(groups,axis=1)), requires_grad=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
       
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_subset,
            kernel_size=(1, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.fc = nn.Linear(50, 17)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / 17))

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
       
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())

        GA = self.GroupA
        GMA = self.GroupMA

        GMA = GMA.detach()
        for g in range(1,9):
            for k in range (1,9):
                GMA[:,:,g,k] = 0
                
        for j in range(17):
            GMA[:,:,14,j] = GMA[:,:,j,14] = 0
            GMA[:,:,13,j] = GMA[:,:,j,13] = 0
        GMA.requires_grad = True

        A = A + GA + GMA      
        norm_learn_A = A.repeat(1,self.out_channels//self.groups,1,1)
        m = x         
        m=self.conv(m)
        n, kc, t, v = m.size()
        A_final=torch.zeros([N,self.num_subset,self.out_channels,17,17],dtype=torch.float,device='cuda').detach()
        for i in range(self.num_subset):
            A_final[:,i,:,:,:] = 0.04 + norm_learn_A[i]

        m = m.view(n, self.num_subset, kc// self.num_subset, t, v)
        m = torch.einsum('nkctv,nkcvw->nctw', (m, A_final))

        # S-ACM
        q = self.avg_pool(m)
        q = self.conv2(q.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) 
        q = self.sigmoid(q)
        q = m + m  *  q.expand_as(m)

        q = self.bn(q)
        q += self.down(x) 
        #end
        return self.relu(q)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, Frames, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = ALSGCN(in_channels, out_channels, A)
        self.tcn1 = MSTFCN(out_channels, out_channels, Frames, kernel_size=5, stride=stride, dilations=[1,2],
                                            residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn_skip(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=17, num_person=2, graph=None, graph_7=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph()

        A = self.graph.A
        self.A_vector = self.get_A(graph, 8)
        self.num_point = num_point        
        self.data_bn = nn.BatchNorm1d(num_person * 80 * num_point)        
        self.to_joint_embedding = nn.Linear(in_channels, 80)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 80))


        self.l1 = TCN_GCN_unit(80, 80, A, 64, residual=False)
        self.l2 = TCN_GCN_unit(80, 80, A, 64)
        self.l3 = TCN_GCN_unit(80, 80, A, 64)
        self.l4 = TCN_GCN_unit(80, 80, A, 64)
        self.l5 = TCN_GCN_unit(80, 160, A, 32, stride=2)
        self.l6 = TCN_GCN_unit(160, 160, A, 32)
        self.l7 = TCN_GCN_unit(160, 160, A, 32)
        self.l8 = TCN_GCN_unit(160, 320, A, 16, stride=2)
        self.l9 = TCN_GCN_unit(320, 320, A, 16)
        self.l10 = TCN_GCN_unit(320, 320, A, 16)

        self.fc = nn.Linear(320, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        # Jump Model
        self.first_tram = nn.Sequential(
                nn.AvgPool2d((4,1)),
                nn.Conv2d(80, 320, 1),
                nn.BatchNorm2d(320),
                nn.ReLU()
            )
        self.second_tram = nn.Sequential(
                nn.AvgPool2d((2,1)),
                nn.Conv2d(160, 320, 1),
                nn.BatchNorm2d(320),
                nn.ReLU()
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        self.num_class=num_class

        #end_code
    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))   


    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        p = self.A_vector
        p = torch.tensor(p,dtype=torch.float)
        x = p.to(x.device).expand(N*M*T, -1, -1) @ x
        
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x2=x
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x3=x
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        x2=self.first_tram(x2)
        x3=self.second_tram(x3)
        x = x + x2 +x3

        x = x.reshape(N, M, 320, -1)
        x = x.mean(3).mean(1)
        return self.fc(x)





