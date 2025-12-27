import torch
import torch.nn as nn
import copy
# from backbone.vision_transformer_tsp import Attention,Block
from timm.models.layers import trunc_normal_, DropPath
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def get_segmented_positional_encoding(seq_len, d_model, segment_len):
    segment_len = seq_len//segment_len
    scale = seq_len//segment_len
    pos_encoding = torch.zeros((seq_len, d_model))
    for pos in range(seq_len):
        segment_index = (pos // segment_len) * (seq_len // segment_len)
        for i in range(0, d_model, 2):
            pos_encoding[pos, i] = torch.sin(torch.tensor(segment_index*scale / (30 ** (2 * i / d_model))))
            if i + 1 < d_model:
                pos_encoding[pos, i + 1] = torch.cos(torch.tensor(segment_index*scale / (30 ** (2 * i / d_model))))
    return pos_encoding.view(1,pos_encoding.shape[0],pos_encoding.shape[1])
def get_segmented_positional_encoding_2(seq_len, d_model, segment_len):
    # Ensure segment_len is valid for the given seq_len
    if seq_len <= 4:
        raise ValueError("Sequence length should be greater than 4.")

    # Initial segment for the first 4 positions
    initial_segment_len = 4
    remaining_len = seq_len - initial_segment_len

    # Calculate number of full segments and the length of the last segment
    num_segments = remaining_len // segment_len
    last_segment_len = remaining_len % segment_len

    pos_encoding = torch.zeros((seq_len, d_model))

    # Handle the initial segment
    for pos in range(initial_segment_len):
        for i in range(0, d_model, 2):
            pos_encoding[pos, i] = torch.sin(torch.tensor(pos / (10000 ** (1 * i / d_model))))
            if i + 1 < d_model:
                pos_encoding[pos, i + 1] = torch.cos(torch.tensor(pos / (10000 ** (1 * i / d_model))))

    # Handle the remaining full segments
    for seg in range(num_segments):
        for pos in range(initial_segment_len + seg * segment_len, initial_segment_len + (seg + 1) * segment_len):
            segment_index = (seg+1) * segment_len
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = torch.sin(torch.tensor(segment_index / (10000 ** (1 * i / d_model))))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = torch.cos(torch.tensor(segment_index / (10000 ** (1 * i / d_model))))

    # Handle the last segment if it exists
    if last_segment_len > 0:
        start_pos = initial_segment_len + num_segments * segment_len
        for pos in range(start_pos, seq_len):
            segment_index = num_segments * segment_len
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = torch.sin(torch.tensor(segment_index / (10000 ** (1 * i / d_model))))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = torch.cos(torch.tensor(segment_index / (10000 ** (1 * i / d_model))))

    return pos_encoding.view(1, pos_encoding.shape[0], pos_encoding.shape[1])
#SPA
class SPAprompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768,prompt_width=4):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        print(self.n_tasks)
        
        # self.posiembedding = get_segmented_positional_encoding(self.e_pool_size+4,768,self.n_tasks).cuda()
        self.posiembedding = get_segmented_positional_encoding_2(self.e_pool_size+4,768,int(self.e_pool_size//self.n_tasks)).cuda()
# dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.
        # self.attn = Attention(
        #     emb_d, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0)
        # self.block = Block(
        #         dim=emb_d, num_heads=2, mlp_ratio=1, qkv_bias=False,
        #         drop=0, attn_drop=0
        #         )
          # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=emb_d, num_heads=2, mlp_ratio=1, qkv_bias=False,
                drop=0, attn_drop=0
            )
            for i in range(4)])
        self.prompt_width = prompt_width
        self.fc = nn.Linear(emb_d, emb_d*2)
        self.cls_token = nn.Parameter(torch.zeros(1,self.prompt_width, emb_d))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.e_pool_size + self.prompt_width, emb_d))
        self.pos_drop = nn.Dropout(p=0.01)
        # trunc_normal_(self.posiembedding, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full parameters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = self.tensor_prompt(1, self.e_pool_size, emb_d)
            # k = self.tensor_prompt(self.e_pool_size, self.key_d)
            # a = self.tensor_prompt(self.e_pool_size, self.key_d)
            # p = self.gram_schmidt(p)
            # k = self.gram_schmidt(k)
            # a = self.gram_schmidt(a)
            
            ## 这里是原始的prompt
            setattr(self, f'e_p_{e}',p)
            # setattr(self, f'e_k_{e}',k)
            # setattr(self, f'e_a_{e}',a)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [8,9,10,11]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            # K = getattr(self,f'e_k_{e}')
            # A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            # k = self.gram_schmidt(K)
            # a = self.gram_schmidt(A)
            # p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',P)
            # setattr(self, f'e_k_{e}',k)
            # setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, l, x_block,sigma,old=False, train=False,reset_fix=False):
        # e prompts
        e_valid = False
        
        if l in self.e_layers:
            e_valid = True
            # B, C = x_querry.shape
            # K = getattr(self,f'e_k_{l}')
            # A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            # print(p.shape)            
            # freeze/control past tasks
            # if train:
            
            ##挑选当前task对应的prompt
            if self.task_count > 0:
                # K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                # A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                p = torch.cat((p[:,:s].detach().clone(),p[:,s:f]), dim=1)
            else:
                # K = K[s:f]
                # A = A[s:f]
                p = p[:,s:f]
            # print(p.shape)
            
            ## 实际上是prompt_token
            cls_tokens = self.cls_token  # stole cls_tokens impl from Phil Wang, thanks
            # print(cls_tokens.shape)
            # print(p.shape)
            p = torch.cat((cls_tokens, p), dim=1)
            # p = p + self.pos_embed[:,:p.size(0),:]
            p = p + self.posiembedding[:,:p.size(1),:]
            p = self.pos_drop(p)
            depth = self.e_layers[0]-7
            for idx, blk in enumerate(self.blocks):  
                if idx+8 <= l:
                    p = blk(p)    
                else:
                    break      
            
            p_emb = p[:,0:self.prompt_width].view(-1, self.emb_d)
            p_emb = self.fc(p_emb)
            p_emb = p_emb.view(-1, self.prompt_width, self.emb_d*2)

            i = int(self.emb_d)
            Ek = p_emb[:,:,:i]
            Ev = p_emb[:,:,i:]
                
            # #Ek,Ev第一个维度复制到batchsize
            # if old == False:
            #     Ek = Ek.repeat(x_block.shape[0],1,1)
            #     Ev = Ev.repeat(x_block.shape[0],1,1)
            # else:
            #     Ek = getattr(self,f'fix_ek_{l}')
            #     Ev = getattr(self,f'fix_ev_{l}')
            #     Ek = Ek.repeat(x_block.shape[0],1,1)
            #     Ev = Ev.repeat(x_block.shape[0],1,1)
            if reset_fix == True:
                if sigma != 0:
                    Ek = getattr(self,f'fix_ek_{l}') * sigma + Ek * (1-sigma)
                    Ev = getattr(self,f'fix_ev_{l}') * sigma + Ev * (1-sigma)
                    setattr(self, f'fix_ek_{l}',Ek)
                    setattr(self, f'fix_ev_{l}',Ev)
                else:
                    setattr(self, f'fix_ek_{l}',Ek)
                    setattr(self, f'fix_ev_{l}',Ev)
            # if sigma != 0: ##sigma等于0时应当调用fix
            #     # print(self.task_count)
            #         Ek = getattr(self,f'fix_ek_{l}') + Ek * sigma
            #         Ev = getattr(self,f'fix_ev_{l}') + Ev * sigma
            # else:
            if (old == False) & (sigma == 0):
                Ek = Ek.repeat(x_block.shape[0],1,1)
                Ev = Ev.repeat(x_block.shape[0],1,1)
            else:
                # print(old,sigma)
                # if l == 8:
                #     print(Ek[0])
                Ek = getattr(self,f'fix_ek_{l}') * sigma + Ek * (1-sigma)
                Ev = getattr(self,f'fix_ev_{l}') * sigma + Ev * (1-sigma)
                Ek = Ek.repeat(x_block.shape[0],1,1)
                Ev = Ev.repeat(x_block.shape[0],1,1)
                # if l == 8:
                #     print(Ek[0])

            loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

    def ortho_penalty(self, t):
        return ((t @t.T - torch.eye(t.shape[0]))**2).mean()
    
    def tensor_prompt(self, a, b, c=None, ortho=False):
        if c is None:
            p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
        if ortho:
            nn.init.orthogonal_(p)
        else:
            nn.init.uniform_(p)
        return p    

