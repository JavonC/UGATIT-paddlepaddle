import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Conv2D, Linear, Sequential



class ResnetGenerator(dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        assert(n_blocks >= 0)
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2d(3), 
                    Conv2D(input_nc, ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                    InstanceNorm(),
                    ReLU()]
        
        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [ReflectionPad2d(1),
                        Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                        InstanceNorm(),
                        ReLU()]
        
        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = ReLU()

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False),
                ReLU(),
                Linear(ngf * mult, ngf * mult, bias_attr=False),
                ReLU()]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False),
                ReLU(),
                Linear(ngf * mult, ngf * mult, bias_attr=False),
                ReLU()]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))
       
        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [Upsample(scale_factor=2, mode='nearest'), 
                        ReflectionPad2d(1),
                        Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                        ILN(int(ngf * mult / 2)),
                        ReLU()]
        
        UpBlock2 += [ReflectionPad2d(3),
                    Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False),
                    Tanh()]
        
        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
       
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        # print('#'*100)  #
        # print('ResnetGenerator')    #
        # print('#'*100)  #
        # print('input', input.shape) #
        x = self.DownBlock(input)
        # print('after DownBlock:',x.shape) #
        gap = fluid.layers.adaptive_pool2d(x, 1, pool_type="avg")
        # print('gap:', gap.shape) #
        gap_bs = fluid.layers.reshape(gap, (x.shape[0], -1))
        # print('gap.view:',gap_bs.shape) #
        gap_logit =self.gap_fc(gap_bs)
        # print('gap_logit:', gap_logit.shape) #
        gap_weight = self.gap_fc.parameters()[0]
        gap_weight = fluid.layers.reshape(gap_weight, (x.shape[0], -1))   
        # print('gap_weight:', gap_weight.shape) #
        gap_weight = fluid.layers.unsqueeze(gap_weight,[2,3])
        
        gap = x * gap_weight
        # print('gap:', gap.shape) #

        gmp = fluid.layers.adaptive_pool2d(x, 1, pool_type="max")
        # print('gmp:', gmp.shape) #
        gmp_bs = fluid.layers.reshape(gmp, (x.shape[0], -1))
        # print('gmp.view:',gmp_bs.shape) #
        gmp_logit = self.gmp_fc(gmp_bs)
        # print('gmp_logit:', gmp_logit.shape) #
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.reshape(gmp_weight, (x.shape[0], -1)) 
        # print('gmp_weight:', gmp_weight.shape) #
        gmp_weight = fluid.layers.unsqueeze(gmp_weight,[2,3])
      
        gmp = x * gmp_weight
        # print('gmp:', gmp.shape) #
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        # print('cam_logit:', cam_logit.shape) #
        x = fluid.layers.concat([gap, gmp], 1)
        # print('concat:', x.shape) #
        x = self.conv1x1(x)
        # print('conv1x1:', x.shape) #
        x = self.relu(x)
        # print('relu:', x.shape) #

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        # print('heatmap:', heatmap.shape) #

        if self.light:
            x_ = fluid.layers.adaptive_pool2d(x, 1, pool_type="max")
            bs = fluid.layers.reshape(x, (x.shape[0], -1))
            x_ = self.FC(bs)
        else:
            bs = fluid.layers.reshape(x, (x.shape[0], -1))
            x_ = self.FC(bs)
        gamma, beta = self.gamma(x_), self.beta(x_)
        # print('gamma:', gamma.shape) #
        # print('beta:', beta.shape) #

        for i in range(self.n_blocks):
            # print('UpBlock1_:'+ str(i+1)) #
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
    
        out = self.UpBlock2(x)
        # print('out:', out.shape) #
        return out, cam_logit, heatmap


class ResnetBlock(dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                    Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                    InstanceNorm(),
                    ReLU()]
        
        conv_block += [ReflectionPad2d(1),
                    Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                    InstanceNorm()]
        
        self.conv_block = Sequential(*conv_block)

    def forward(self, input):
        out = input + self.conv_block(input)

        return out


class ResnetAdaILNBlock(dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU()

        self.pad2 = ReflectionPad2d(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, input, gamma, beta):
        # print('ResnetAdaILN')
        # print('input:', input.shape) #
        x = self.pad1(input)
        # print('pad1:', x.shape) #
        x = self.conv1(x)
        # print('conv1:', x.shape) #
        x = self.norm1(x, gamma, beta)
        # print('norm1:', x.shape) #
        x = self.relu1(x)
        # print('relu1:', x.shape) #
        x = self.pad2(x)
        # print('pad2:', x.shape) #
        x = self.conv2(x)
        # print('conv2:', x.shape) #
        out = self.norm2(x, gamma, beta)
        # print('norm2:', x.shape) #
        # print('out:',(out+input).shape) #

        return out + input


class adaILN(dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',default_initializer=fluid.initializer.ConstantInitializer(0.9))
       
    def forward(self, input, gamma, beta):
        # print('AdaILN')   #
        # print('input:', input.shape) #
        in_mean = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True)
        # print('in_mean:', in_mean.shape) #
        in_var = fluid.layers.reduce_mean((input - in_mean)**2, dim=[2,3], keep_dim=True)
        # print('in_var:', in_var.shape) #
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        # print('out_in:', out_in.shape) #
        ln_mean = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True)
        # print('ln_mean:', ln_mean.shape) #
        ln_var = fluid.layers.reduce_mean((input - ln_mean)**2, dim=[1,2,3], keep_dim=True)
        # print('ln_var:', ln_var.shape) #
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        # print('out_ln:', out_ln.shape) #
        rho_expand = fluid.layers.expand(self.rho,(input.shape[0],1,1,1))
        # print('rho_expand:', rho_expand.shape) #
        out = rho_expand * out_in + (1-rho_expand) * out_ln
        # print('out:', out.shape) #
        
        gamma = fluid.layers.unsqueeze(gamma, [2,3])
        beta = fluid.layers.unsqueeze(beta,[2,3])

        out = out * gamma + beta
        # print('gamma:', gamma.shape) #
        # print('beta:', beta.shape) #
        # print('out:', out.shape) #

        return out


class ILN(dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',default_initializer=fluid.initializer.ConstantInitializer(0.0))
        self.gamma = self.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',default_initializer=fluid.initializer.ConstantInitializer(1.0))
        self.beta = self.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',default_initializer=fluid.initializer.ConstantInitializer(0.0))
        
        
    def forward(self, input):
        in_mean = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True)
        in_var = fluid.layers.reduce_mean((input - in_mean) ** 2, dim=[2,3], keep_dim=True)
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)

        ln_mean = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True)
        ln_var = fluid.layers.reduce_mean((input - ln_mean) ** 2, dim=[1,2,3], keep_dim=True)
        
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        rho_expand = fluid.layers.expand(self.rho, (input.shape[0],1,1,1))
        gamma_expand = fluid.layers.expand(self.gamma, (input.shape[0],1,1,1))
        beta_expand = fluid.layers.expand(self.beta, (input.shape[0],1,1,1))
        out = rho_expand * out_in + (1-rho_expand) * out_ln
        out = out * gamma_expand + beta_expand

        return out


class ReLU(dygraph.Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        return fluid.layers.relu(input)


class LeakyReLu(dygraph.Layer):
    def __init__(self, alpha):
        super(LeakyReLu, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return fluid.layers.leaky_relu(input, alpha=self.alpha)


class Tanh(dygraph.Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        return fluid.layers.tanh(input)
    

class ReflectionPad2d(dygraph.Layer):
    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, input):
        return fluid.layers.pad2d(input=input, paddings=self.padding, mode='reflect')

class InstanceNorm(dygraph.Layer):
    def __init__(self):
        super(InstanceNorm, self).__init__()
    
    def forward(self, input):
        return fluid.layers.instance_norm(input=input)

class Upsample(dygraph.Layer):
    def __init__(self, scale_factor = 2, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, input):
        if self.mode == 'nearest':
            out = fluid.layers.resize_nearest(input, scale=self.scale_factor)
        elif self.mode == 'bilinear':
            out = fluid.layers.resize_bilinear(input, scale=self.scale_factor)
        elif self.mode == 'trilinear':
            out = fluid.layers.resize_trilinear(input, scale=self.scale_factor)
        return out


class Spectralnorm(dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = fluid.dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


class Discriminator(dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2d(1),
                Spectralnorm(
                    Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True)
                ),
                LeakyReLu(0.2)]
        
        for i in range(1, n_layers -2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(1),
                    Spectralnorm(
                        Conv2D(ndf * mult, ndf * mult *2, filter_size=4, stride=2, padding=0, bias_attr=True)
                    ),
                    LeakyReLu(0.2)]
        
        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(1),
                Spectralnorm(
                    Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True)
                ),
                LeakyReLu(0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers -2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu = LeakyReLu(0.2)

        self.pad = ReflectionPad2d(1)
        self.conv = Spectralnorm(
            Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False)
        )
        self.model = Sequential(*model)
    
    def forward(self, input):
        # print('#'*100)  #
        # print('Discriminator')  #
        # print('#'*100)  #
        # print('input', input.shape)  #
        x = self.model(input)
        # print('model', x.shape)  #
        gap = fluid.layers.adaptive_pool2d(x, 1, pool_type="avg")
        # print('gap', gap.shape)  #
        gap_bs = fluid.layers.reshape(gap, (x.shape[0], -1))
        gap_logit =self.gap_fc(gap_bs)
        # print('gap_logit', gap_logit.shape)  #
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.reshape(gap_weight, (x.shape[0], -1))
        gap_weight = fluid.layers.unsqueeze(gap_weight,[2,3])
        # print('gap_weight', gap_weight.shape)  #
        gap = x * gap_weight
        # print('gap', gap.shape)  #
        gmp = fluid.layers.adaptive_pool2d(x, 1, pool_type="max")
        # print('gmp', gmp.shape)  #
        gmp_bs = fluid.layers.reshape(gmp, (x.shape[0], -1))
        gmp_logit = self.gmp_fc(gmp_bs)
        # print('gmp_logit', gmp_logit.shape)  #
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.reshape(gmp_weight, (x.shape[0], -1))
        gmp_weight = fluid.layers.unsqueeze(gmp_weight,[2,3])
        # print('gmp_weight', gmp_weight.shape)  #

        gmp = x * gmp_weight
        # print('gmp', gmp.shape)  #
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        # print('cam_logit', cam_logit.shape)  #
        x = fluid.layers.concat([gap, gmp], 1)
        # print('concat', x.shape)  #
        x = self.conv1x1(x)
        # print('conv1x1', x.shape)  #
        x = self.leaky_relu(x)
        # print('leaky_relu', x.shape)  #

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        # print('heatmap', heatmap.shape)  #

        x = self.pad(x)
        # print('pad', x.shape)  #
        out = self.conv(x)
        # print('conv', out.shape)  #

        return out, cam_logit, heatmap

def clip_rho(net, vmin=0, vmax=1):
    for name, param in net.named_parameters():
        if 'rho' in name:
            param.set_value(fluid.layers.clip(param, vmin, vmax))
