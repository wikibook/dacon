import math
import torch 
import torch.nn as nn
"""
code for mlp with skip connection model.
scalable model for ensemble.
"""

##############################
###### Activation ############
##############################

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.init_weights()

    def init_weights(self):
        self.weight.data.fill_(1.0)
        self.bias.data.zero_()

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

##################################
######## Free format #############
##################################
"""
- Baseline :  
     - MLP(ANN) 구성
     - Skip Connection idea 적용 => 이전 정보를 효율적으로 활용
     - LayerNorm : 블록마다 feature Normalization 사용하여 수렴을 촉진 
     - GELU 활성화 함수 적용 (미분 가능 및 음수 값에 대한 계산 확대)
"""
class SkipConnectionModel(nn.Module):
    """
    >> model = Model(f_in, f_out, 300, 2000, 4000, 7000, 10000)
    300, 2000, 4000, 7000, 10000 : channels
    """
    def __init__(self, fn_in=226, fn_out=4, *args):        
        super(SkipConnectionModel, self).__init__()
        self.ln = LayerNorm(10000) #10000
        self.ln1 = LayerNorm(7000) # 7000
        self.ln2 = LayerNorm(4000) # 4000
        self.ln3 = LayerNorm(2000) # 2000
        
        self.upblock1 = nn.Sequential(nn.Linear(fn_in, 2000),GELU(),nn.BatchNorm1d(2000)) # 2000
        self.upblock2 = nn.Sequential(nn.Linear(2000,4000),GELU(),nn.BatchNorm1d(4000)) # 4000
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000)) # 7000
        self.upblock4 = nn.Sequential(nn.Linear(7000,10000),GELU(),nn.BatchNorm1d(10000)) # 10000

        self.downblock1 = nn.Sequential(nn.Linear(10000, 7000),GELU(),nn.BatchNorm1d(7000)) #7000
        self.downblock2 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000)) # 4000
        self.downblock3 = nn.Sequential(nn.Linear(4000, 2000),GELU(),nn.BatchNorm1d(2000)) # 2000
        self.downblock4 = nn.Sequential(nn.Linear(2000, 300),GELU(),nn.BatchNorm1d(300)) # 10000
        
        self.fclayer = nn.Sequential(nn.Linear(300, fn_out))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        
        downblock1_out = self.downblock1(self.ln(upblock4_out)) # 계층 정규화 적용.
        skipblock1 = downblock1_out + upblock3_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        
        output = self.fclayer(downblock4_out)
        
        return output

########################################
########################################

"""
- Test Models for Ensemble
"""

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
#         self.ln = LayerNorm(1args[4]0)
        self.ln = LayerNorm(10000)
        self.ln1 = LayerNorm(7000)
        self.ln2 = LayerNorm(4000)
        self.ln3 = LayerNorm(2000)
        
        self.upblock1 = nn.Sequential(nn.Linear(226, 2000),GELU(),nn.BatchNorm1d(2000))
        self.upblock2 = nn.Sequential(nn.Linear(2000,4000),GELU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,10000),GELU(),nn.BatchNorm1d(10000))
        #self.upblock5 = nn.Sequential(nn.Linear(10000,13000),GELU(),nn.BatchNorm1d(13000))

        #self.downblock1 = nn.Sequential(nn.Linear(13000, 10000),GELU(),nn.BatchNorm1d(10000))
        self.downblock1 = nn.Sequential(nn.Linear(10000, 7000),GELU(),nn.BatchNorm1d(7000))
        self.downblock2 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000))
        self.downblock3 = nn.Sequential(nn.Linear(4000, 2000),GELU(),nn.BatchNorm1d(2000))
        self.downblock4 = nn.Sequential(nn.Linear(2000, 300),GELU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        #upblock5_out = self.upblock5(upblock4_out)
        
        downblock1_out = self.downblock1(self.ln(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        
        output = self.fclayer(downblock4_out)
        
        return output


class TestModel1(nn.Module):
    def __init__(self):
        super(TestModel1, self).__init__()
        
#         self.ln = LayerNorm(13000)
        self.ln = LayerNorm(10000)
        self.ln1 = LayerNorm(7000)
        self.ln2 = LayerNorm(4000)
        self.ln3 = LayerNorm(1000)
        
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),GELU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,4000),GELU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,10000),GELU(),nn.BatchNorm1d(10000))
        #self.upblock5 = nn.Sequential(nn.Linear(10000,13000),GELU(),nn.BatchNorm1d(13000))

        #self.downblock1 = nn.Sequential(nn.Linear(13000, 10000),GELU(),nn.BatchNorm1d(10000))
        self.downblock1 = nn.Sequential(nn.Linear(10000, 7000),GELU(),nn.BatchNorm1d(7000))
        self.downblock2 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000))
        self.downblock3 = nn.Sequential(nn.Linear(4000, 1000),GELU(),nn.BatchNorm1d(1000))
        self.downblock4 = nn.Sequential(nn.Linear(1000, 300),GELU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        #upblock5_out = self.upblock5(upblock4_out)
        
        downblock1_out = self.downblock1(self.ln(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        
        output = self.fclayer(downblock4_out)
        
        return output
    
# Model 2

class TestModel2(nn.Module):
    def __init__(self):
        super(TestModel2, self).__init__()
        
#         self.ln = LayerNorm(13000)
        self.ln = LayerNorm(20000)
        self.ln1 = LayerNorm(13000)
        self.ln2 = LayerNorm(7000)
        self.ln3 = LayerNorm(4000)
        self.ln4 = LayerNorm(1000)
        self.ln5 = LayerNorm(13000)
       
        
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,4000),nn.ReLU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,13000),nn.ReLU(),nn.BatchNorm1d(13000))
        self.upblock5 = nn.Sequential(nn.Linear(13000,20000),nn.ReLU(),nn.BatchNorm1d(20000))
        self.upblock6 = nn.Sequential(nn.Linear(20000,13000),nn.ReLU(),nn.BatchNorm1d(13000))

        self.downblock1 = nn.Sequential(nn.Linear(13000, 20000),nn.ReLU(),nn.BatchNorm1d(20000))
        self.downblock2 = nn.Sequential(nn.Linear(20000, 13000),nn.ReLU(),nn.BatchNorm1d(13000))
        self.downblock3 = nn.Sequential(nn.Linear(13000, 7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.downblock4 = nn.Sequential(nn.Linear(7000, 4000),nn.ReLU(),nn.BatchNorm1d(4000))
        self.downblock5 = nn.Sequential(nn.Linear(4000, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.downblock6 = nn.Sequential(nn.Linear(1000, 300),nn.ReLU(),nn.BatchNorm1d(300))
        
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        upblock5_out = self.upblock5(upblock4_out)
        upblock6_out = self.upblock6(upblock5_out)

        
        downblock1_out = self.downblock1(self.ln1(upblock6_out))
        skipblock1 = downblock1_out + upblock5_out  # 20000
        downblock2_out = self.downblock2(self.ln(skipblock1))
        skipblock2 = downblock2_out + upblock4_out # 13000
        downblock3_out = self.downblock3(self.ln5(skipblock2))
        skipblock3 = downblock3_out + upblock3_out # 7000
        downblock4_out = self.downblock4(self.ln2(skipblock3))
        skipblock4 = downblock4_out + upblock2_out # 4000
        
        downblock5_out = self.downblock5(self.ln3(skipblock4))
        skipblock5 = downblock5_out + upblock1_out
        downblock6_out = self.downblock6(self.ln4(skipblock5))
        
        output = self.fclayer(downblock6_out)
        
        return output

# Model3
class TestModel3(nn.Module):
    """
    Model for (20,40)
    """
    def __init__(self):
        super(TestModel3, self).__init__()
        
        self.ln = LayerNorm(17000)
        self.ln1 = LayerNorm(13000)
        self.ln2 = LayerNorm(7000)
        self.ln3 = LayerNorm(5000)        

        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,3000),nn.ReLU(),nn.BatchNorm1d(3000))
        self.upblock3 = nn.Sequential(nn.Linear(3000,5000),nn.ReLU(),nn.BatchNorm1d(5000))
        self.upblock4 = nn.Sequential(nn.Linear(5000,7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.upblock5 = nn.Sequential(nn.Linear(7000,13000),nn.ReLU(),nn.BatchNorm1d(13000))
        self.upblock6 = nn.Sequential(nn.Linear(13000,17000),nn.ReLU(),nn.BatchNorm1d(17000))
        
        self.downblock1 = nn.Sequential(nn.Linear(17000,13000),nn.ReLU(),nn.BatchNorm1d(13000))
        self.downblock2 = nn.Sequential(nn.Linear(13000, 7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.downblock3 = nn.Sequential(nn.Linear(7000, 5000),nn.ReLU(),nn.BatchNorm1d(5000))
        self.downblock4 = nn.Sequential(nn.Linear(5000, 3000),nn.ReLU(),nn.BatchNorm1d(3000))
        self.downblock5 = nn.Sequential(nn.Linear(3000, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.downblock6 = nn.Sequential(nn.Linear(1000, 300),nn.ReLU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        upblock5_out = self.upblock5(upblock4_out)
        upblock6_out = self.upblock6(upblock5_out)
                                    
        downblock1_out = self.dropout(self.downblock1(self.ln(upblock6_out)))
        skipblock1 = downblock1_out + upblock5_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock4_out
        downblock3_out = self.dropout(self.downblock3(self.ln2(skipblock2)))
        skipblock3 = downblock3_out + upblock3_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        skipblock4 = downblock4_out + upblock2_out
        downblock5_out = self.downblock5(skipblock4)
        skipblock5 = self.dropout(downblock5_out + upblock1_out)
        downblock6_out = self.downblock6(skipblock5)
        
        output = self.fclayer(downblock6_out)
        
        return output

class TestModel4(nn.Module):
    def __init__(self):
        super(TestModel4, self).__init__()
        
        self.ln = LayerNorm(10000)
        self.ln1 = LayerNorm(7000)
        self.ln2 = LayerNorm(4000)
        self.ln3 = LayerNorm(1000)
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,10000),nn.ReLU(),nn.BatchNorm1d(10000))
        self.upblock3 = nn.Sequential(nn.Linear(10000,7000), nn.ReLU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,4000),nn.ReLU(),nn.BatchNorm1d(4000))

        self.downblock1 = nn.Sequential(nn.Linear(4000, 7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.downblock2 = nn.Sequential(nn.Linear(7000, 10000),nn.ReLU(),nn.BatchNorm1d(10000))
        self.downblock3 = nn.Sequential(nn.Linear(10000, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.downblock4 = nn.Sequential(nn.Linear(1000, 300),nn.ReLU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.dropout(self.upblock2(upblock1_out))
        upblock3_out = self.dropout(self.upblock3(upblock2_out))
        upblock4_out = self.dropout(self.upblock4(upblock3_out))
        
        downblock1_out = self.downblock1(self.ln2(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out # 7000
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out  # 10000
        downblock3_out = self.downblock3(self.ln(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        
        output = self.fclayer(downblock4_out)
        
        return output
    
class TestModel5(nn.Module):
    def __init__(self):
        super(TestModel5, self).__init__()
        
        self.ln = LayerNorm(13000)
        self.ln1 = LayerNorm(11000)
        self.ln2 = LayerNorm(7000)
        self.ln3 = LayerNorm(4000)
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),GELU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,4000),GELU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,11000),GELU(),nn.BatchNorm1d(11000))
        self.upblock5 = nn.Sequential(nn.Linear(11000,13000),GELU(),nn.BatchNorm1d(13000))

        self.downblock1 = nn.Sequential(nn.Linear(13000, 11000),GELU(),nn.BatchNorm1d(11000))
        self.downblock2 = nn.Sequential(nn.Linear(11000, 7000),GELU(),nn.BatchNorm1d(7000))
        self.downblock3 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000))
        self.downblock4 = nn.Sequential(nn.Linear(4000, 1000),GELU(),nn.BatchNorm1d(1000))
        self.downblock5 = nn.Sequential(nn.Linear(1000, 300),GELU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        upblock5_out = self.upblock5(upblock4_out)
        
        downblock1_out = self.downblock1(self.ln(upblock5_out))
        skipblock1 = downblock1_out + upblock4_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock3_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock2_out
        downblock4_out = self.dropout(self.downblock4(self.ln3(skipblock3)))
        skipblock4 = downblock4_out + upblock1_out
        downblock5_out = self.downblock5(skipblock4)
        
        output = self.fclayer(downblock5_out)
        
        return output

    
class TestModel6(nn.Module):
    def __init__(self):
        super(TestModel6, self).__init__()
        
#         self.ln = LayerNorm(13000)
        self.ln = LayerNorm(10000)
        self.ln1 = LayerNorm(7000)
        self.ln2 = LayerNorm(4000)
        
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),GELU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,4000),GELU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,10000),GELU(),nn.BatchNorm1d(10000))
        #self.upblock5 = nn.Sequential(nn.Linear(10000,13000),GELU(),nn.BatchNorm1d(13000))

        #self.downblock1 = nn.Sequential(nn.Linear(13000, 10000),GELU(),nn.BatchNorm1d(10000))
        self.downblock1 = nn.Sequential(nn.Linear(10000, 7000),GELU(),nn.BatchNorm1d(7000))
        self.downblock2 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000))
        self.downblock3 = nn.Sequential(nn.Linear(4000, 1000),GELU(),nn.BatchNorm1d(1000))
        self.downblock4 = nn.Sequential(nn.Linear(1000, 300),GELU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        #upblock5_out = self.upblock5(upblock4_out)
        
        downblock1_out = self.downblock1(self.ln(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(skipblock3)
        
        output = self.fclayer(downblock4_out)
        
        return output
