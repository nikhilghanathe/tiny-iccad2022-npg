м╢
╤г
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878Бц
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
:
*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:*
dtype0
l

bn_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
bn_1/gamma
e
bn_1/gamma/Read/ReadVariableOpReadVariableOp
bn_1/gamma*
_output_shapes
:*
dtype0
j
	bn_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	bn_1/beta
c
bn_1/beta/Read/ReadVariableOpReadVariableOp	bn_1/beta*
_output_shapes
:*
dtype0
x
bn_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebn_1/moving_mean
q
$bn_1/moving_mean/Read/ReadVariableOpReadVariableOpbn_1/moving_mean*
_output_shapes
:*
dtype0
А
bn_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namebn_1/moving_variance
y
(bn_1/moving_variance/Read/ReadVariableOpReadVariableOpbn_1/moving_variance*
_output_shapes
:*
dtype0
О
dw_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namedw_1/depthwise_kernel
З
)dw_1/depthwise_kernel/Read/ReadVariableOpReadVariableOpdw_1/depthwise_kernel*&
_output_shapes
:*
dtype0
j
	dw_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	dw_1/bias
c
dw_1/bias/Read/ReadVariableOpReadVariableOp	dw_1/bias*
_output_shapes
:*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
z
pw_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namepw_1/kernel
s
pw_1/kernel/Read/ReadVariableOpReadVariableOppw_1/kernel*&
_output_shapes
:*
dtype0
j
	pw_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	pw_1/bias
c
pw_1/bias/Read/ReadVariableOpReadVariableOp	pw_1/bias*
_output_shapes
:*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
О
dw_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namedw_2/depthwise_kernel
З
)dw_2/depthwise_kernel/Read/ReadVariableOpReadVariableOpdw_2/depthwise_kernel*&
_output_shapes
:*
dtype0
j
	dw_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	dw_2/bias
c
dw_2/bias/Read/ReadVariableOpReadVariableOp	dw_2/bias*
_output_shapes
:*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
z
pw_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namepw_2/kernel
s
pw_2/kernel/Read/ReadVariableOpReadVariableOppw_2/kernel*&
_output_shapes
:*
dtype0
j
	pw_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	pw_2/bias
c
pw_2/bias/Read/ReadVariableOpReadVariableOp	pw_2/bias*
_output_shapes
:*
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
Ф
dw_ee_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namedw_ee_1/depthwise_kernel
Н
,dw_ee_1/depthwise_kernel/Read/ReadVariableOpReadVariableOpdw_ee_1/depthwise_kernel*&
_output_shapes
:*
dtype0
p
dw_ee_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedw_ee_1/bias
i
 dw_ee_1/bias/Read/ReadVariableOpReadVariableOpdw_ee_1/bias*
_output_shapes
:*
dtype0
Г
dense_1_ee_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╨
*$
shared_namedense_1_ee_1/kernel
|
'dense_1_ee_1/kernel/Read/ReadVariableOpReadVariableOpdense_1_ee_1/kernel*
_output_shapes
:	╨
*
dtype0
z
dense_1_ee_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namedense_1_ee_1/bias
s
%dense_1_ee_1/bias/Read/ReadVariableOpReadVariableOpdense_1_ee_1/bias*
_output_shapes
:
*
dtype0
В
dense_2_ee_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_namedense_2_ee_1/kernel
{
'dense_2_ee_1/kernel/Read/ReadVariableOpReadVariableOpdense_2_ee_1/kernel*
_output_shapes

:
*
dtype0
z
dense_2_ee_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namedense_2_ee_1/bias
s
%dense_2_ee_1/bias/Read/ReadVariableOpReadVariableOpdense_2_ee_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
аh
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*█g
value╤gB╬g B╟g
с
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
layer-19
layer_with_weights-11
layer-20
layer_with_weights-12
layer-21
regularization_losses
trainable_variables
	variables
	keras_api

signatures
%
#_self_saveable_object_factories
Н

kernel
bias
#_self_saveable_object_factories
 regularization_losses
!trainable_variables
"	variables
#	keras_api
╝
$axis
	%gamma
&beta
'moving_mean
(moving_variance
#)_self_saveable_object_factories
*regularization_losses
+trainable_variables
,	variables
-	keras_api
w
#._self_saveable_object_factories
/regularization_losses
0trainable_variables
1	variables
2	keras_api
w
#3_self_saveable_object_factories
4regularization_losses
5trainable_variables
6	variables
7	keras_api
Ч
8depthwise_kernel
9bias
#:_self_saveable_object_factories
;regularization_losses
<trainable_variables
=	variables
>	keras_api
╝
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
w
#I_self_saveable_object_factories
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
Н

Nkernel
Obias
#P_self_saveable_object_factories
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
╝
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
#Z_self_saveable_object_factories
[regularization_losses
\trainable_variables
]	variables
^	keras_api
w
#__self_saveable_object_factories
`regularization_losses
atrainable_variables
b	variables
c	keras_api
Ч
ddepthwise_kernel
ebias
#f_self_saveable_object_factories
gregularization_losses
htrainable_variables
i	variables
j	keras_api
╝
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
#p_self_saveable_object_factories
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
w
#u_self_saveable_object_factories
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
О

zkernel
{bias
#|_self_saveable_object_factories
}regularization_losses
~trainable_variables
	variables
А	keras_api
╞
	Бaxis

Вgamma
	Гbeta
Дmoving_mean
Еmoving_variance
$Ж_self_saveable_object_factories
Зregularization_losses
Иtrainable_variables
Й	variables
К	keras_api
|
$Л_self_saveable_object_factories
Мregularization_losses
Нtrainable_variables
О	variables
П	keras_api
Ю
Рdepthwise_kernel
	Сbias
$Т_self_saveable_object_factories
Уregularization_losses
Фtrainable_variables
Х	variables
Ц	keras_api
|
$Ч_self_saveable_object_factories
Шregularization_losses
Щtrainable_variables
Ъ	variables
Ы	keras_api
|
$Ь_self_saveable_object_factories
Эregularization_losses
Юtrainable_variables
Я	variables
а	keras_api
Ф
бkernel
	вbias
$г_self_saveable_object_factories
дregularization_losses
еtrainable_variables
ж	variables
з	keras_api
Ф
иkernel
	йbias
$к_self_saveable_object_factories
лregularization_losses
мtrainable_variables
н	variables
о	keras_api
 
╬
0
1
%2
&3
84
95
@6
A7
N8
O9
V10
W11
d12
e13
l14
m15
z16
{17
В18
Г19
Р20
С21
б22
в23
и24
й25
а
0
1
%2
&3
'4
(5
86
97
@8
A9
B10
C11
N12
O13
V14
W15
X16
Y17
d18
e19
l20
m21
n22
o23
z24
{25
В26
Г27
Д28
Е29
Р30
С31
б32
в33
и34
й35
▓
пmetrics
regularization_losses
 ░layer_regularization_losses
▒non_trainable_variables
trainable_variables
▓layer_metrics
│layers
	variables
 
 
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
▓
┤metrics
 regularization_losses
 ╡layer_regularization_losses
╢non_trainable_variables
!trainable_variables
╖layer_metrics
╕layers
"	variables
 
US
VARIABLE_VALUE
bn_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bn_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbn_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

%0
&1

%0
&1
'2
(3
▓
╣metrics
*regularization_losses
 ║layer_regularization_losses
╗non_trainable_variables
+trainable_variables
╝layer_metrics
╜layers
,	variables
 
 
 
 
▓
╛metrics
/regularization_losses
 ┐layer_regularization_losses
└non_trainable_variables
0trainable_variables
┴layer_metrics
┬layers
1	variables
 
 
 
 
▓
├metrics
4regularization_losses
 ─layer_regularization_losses
┼non_trainable_variables
5trainable_variables
╞layer_metrics
╟layers
6	variables
ki
VARIABLE_VALUEdw_1/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	dw_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

80
91

80
91
▓
╚metrics
;regularization_losses
 ╔layer_regularization_losses
╩non_trainable_variables
<trainable_variables
╦layer_metrics
╠layers
=	variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

@0
A1

@0
A1
B2
C3
▓
═metrics
Eregularization_losses
 ╬layer_regularization_losses
╧non_trainable_variables
Ftrainable_variables
╨layer_metrics
╤layers
G	variables
 
 
 
 
▓
╥metrics
Jregularization_losses
 ╙layer_regularization_losses
╘non_trainable_variables
Ktrainable_variables
╒layer_metrics
╓layers
L	variables
WU
VARIABLE_VALUEpw_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	pw_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

N0
O1

N0
O1
▓
╫metrics
Qregularization_losses
 ╪layer_regularization_losses
┘non_trainable_variables
Rtrainable_variables
┌layer_metrics
█layers
S	variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

V0
W1

V0
W1
X2
Y3
▓
▄metrics
[regularization_losses
 ▌layer_regularization_losses
▐non_trainable_variables
\trainable_variables
▀layer_metrics
рlayers
]	variables
 
 
 
 
▓
сmetrics
`regularization_losses
 тlayer_regularization_losses
уnon_trainable_variables
atrainable_variables
фlayer_metrics
хlayers
b	variables
ki
VARIABLE_VALUEdw_2/depthwise_kernel@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	dw_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

d0
e1

d0
e1
▓
цmetrics
gregularization_losses
 чlayer_regularization_losses
шnon_trainable_variables
htrainable_variables
щlayer_metrics
ъlayers
i	variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

l0
m1

l0
m1
n2
o3
▓
ыmetrics
qregularization_losses
 ьlayer_regularization_losses
эnon_trainable_variables
rtrainable_variables
юlayer_metrics
яlayers
s	variables
 
 
 
 
▓
Ёmetrics
vregularization_losses
 ёlayer_regularization_losses
Єnon_trainable_variables
wtrainable_variables
єlayer_metrics
Їlayers
x	variables
WU
VARIABLE_VALUEpw_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	pw_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

z0
{1

z0
{1
▓
їmetrics
}regularization_losses
 Ўlayer_regularization_losses
ўnon_trainable_variables
~trainable_variables
°layer_metrics
∙layers
	variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

В0
Г1
 
В0
Г1
Д2
Е3
╡
·metrics
Зregularization_losses
 √layer_regularization_losses
№non_trainable_variables
Иtrainable_variables
¤layer_metrics
■layers
Й	variables
 
 
 
 
╡
 metrics
Мregularization_losses
 Аlayer_regularization_losses
Бnon_trainable_variables
Нtrainable_variables
Вlayer_metrics
Гlayers
О	variables
om
VARIABLE_VALUEdw_ee_1/depthwise_kernelAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdw_ee_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Р0
С1

Р0
С1
╡
Дmetrics
Уregularization_losses
 Еlayer_regularization_losses
Жnon_trainable_variables
Фtrainable_variables
Зlayer_metrics
Иlayers
Х	variables
 
 
 
 
╡
Йmetrics
Шregularization_losses
 Кlayer_regularization_losses
Лnon_trainable_variables
Щtrainable_variables
Мlayer_metrics
Нlayers
Ъ	variables
 
 
 
 
╡
Оmetrics
Эregularization_losses
 Пlayer_regularization_losses
Рnon_trainable_variables
Юtrainable_variables
Сlayer_metrics
Тlayers
Я	variables
`^
VARIABLE_VALUEdense_1_ee_1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_1_ee_1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

б0
в1

б0
в1
╡
Уmetrics
дregularization_losses
 Фlayer_regularization_losses
Хnon_trainable_variables
еtrainable_variables
Цlayer_metrics
Чlayers
ж	variables
`^
VARIABLE_VALUEdense_2_ee_1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_2_ee_1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

и0
й1

и0
й1
╡
Шmetrics
лregularization_losses
 Щlayer_regularization_losses
Ъnon_trainable_variables
мtrainable_variables
Ыlayer_metrics
Ьlayers
н	variables
 
 
H
'0
(1
B2
C3
X4
Y5
n6
o7
Д8
Е9
 
ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 
 
 
 
 
 

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

B0
C1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

X0
Y1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

n0
o1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Д0
Е1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
М
serving_default_input_1Placeholder*0
_output_shapes
:         т	*
dtype0*%
shape:         т	
─	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_1/kernelconv_1/bias
bn_1/gamma	bn_1/betabn_1/moving_meanbn_1/moving_variancedw_1/depthwise_kernel	dw_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancepw_1/kernel	pw_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedw_2/depthwise_kernel	dw_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancepw_2/kernel	pw_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedw_ee_1/depthwise_kerneldw_ee_1/biasdense_1_ee_1/kerneldense_1_ee_1/biasdense_2_ee_1/kerneldense_2_ee_1/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_13883
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┘
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOpbn_1/gamma/Read/ReadVariableOpbn_1/beta/Read/ReadVariableOp$bn_1/moving_mean/Read/ReadVariableOp(bn_1/moving_variance/Read/ReadVariableOp)dw_1/depthwise_kernel/Read/ReadVariableOpdw_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOppw_1/kernel/Read/ReadVariableOppw_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp)dw_2/depthwise_kernel/Read/ReadVariableOpdw_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOppw_2/kernel/Read/ReadVariableOppw_2/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp,dw_ee_1/depthwise_kernel/Read/ReadVariableOp dw_ee_1/bias/Read/ReadVariableOp'dense_1_ee_1/kernel/Read/ReadVariableOp%dense_1_ee_1/bias/Read/ReadVariableOp'dense_2_ee_1/kernel/Read/ReadVariableOp%dense_2_ee_1/bias/Read/ReadVariableOpConst*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_15395
Д	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/bias
bn_1/gamma	bn_1/betabn_1/moving_meanbn_1/moving_variancedw_1/depthwise_kernel	dw_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancepw_1/kernel	pw_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedw_2/depthwise_kernel	dw_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancepw_2/kernel	pw_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedw_ee_1/depthwise_kerneldw_ee_1/biasdense_1_ee_1/kerneldense_1_ee_1/biasdense_2_ee_1/kerneldense_2_ee_1/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_15513РР
╨
з
?__inference_pw_1_layer_call_and_return_conditional_losses_12882

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2	
BiasAdd╗
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╣:::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Ц
З
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12241

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Э
и
5__inference_batch_normalization_3_layer_call_fn_15106

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_125762
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╙
Й
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15144

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣:::::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
ЦЬ
Я
G__inference_functional_1_layer_call_and_return_conditional_losses_14216

inputs)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource 
bn_1_readvariableop_resource"
bn_1_readvariableop_1_resource1
-bn_1_fusedbatchnormv3_readvariableop_resource3
/bn_1_fusedbatchnormv3_readvariableop_1_resource*
&dw_1_depthwise_readvariableop_resource(
$dw_1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource'
#pw_1_conv2d_readvariableop_resource(
$pw_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
&dw_2_depthwise_readvariableop_resource(
$dw_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource'
#pw_2_conv2d_readvariableop_resource(
$pw_2_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource-
)dw_ee_1_depthwise_readvariableop_resource+
'dw_ee_1_biasadd_readvariableop_resource/
+dense_1_ee_1_matmul_readvariableop_resource0
,dense_1_ee_1_biasadd_readvariableop_resource/
+dense_2_ee_1_matmul_readvariableop_resource0
,dense_2_ee_1_biasadd_readvariableop_resource
identityИк
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOp╣
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё*
paddingSAME*
strides
2
conv_1/Conv2Dб
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOpе
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё2
conv_1/BiasAddГ
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
:*
dtype02
bn_1/ReadVariableOpЙ
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn_1/ReadVariableOp_1╢
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp╝
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1·
bn_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ё:::::*
epsilon%oГ:*
is_training( 2
bn_1/FusedBatchNormV3А
activation/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ё2
activation/ReluК
dropout/IdentityIdentityactivation/Relu:activations:0*
T0*0
_output_shapes
:         ё2
dropout/Identityн
dw_1/depthwise/ReadVariableOpReadVariableOp&dw_1_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
dw_1/depthwise/ReadVariableOpЕ
dw_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dw_1/depthwise/ShapeН
dw_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dw_1/depthwise/dilation_rate▐
dw_1/depthwiseDepthwiseConv2dNativedropout/Identity:output:0%dw_1/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
dw_1/depthwiseЫ
dw_1/BiasAdd/ReadVariableOpReadVariableOp$dw_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dw_1/BiasAdd/ReadVariableOpа
dw_1/BiasAddBiasAdddw_1/depthwise:output:0#dw_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
dw_1/BiasAdd░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1╥
$batch_normalization/FusedBatchNormV3FusedBatchNormV3dw_1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3У
activation_1/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2
activation_1/Reluд
pw_1/Conv2D/ReadVariableOpReadVariableOp#pw_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
pw_1/Conv2D/ReadVariableOp╠
pw_1/Conv2DConv2Dactivation_1/Relu:activations:0"pw_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
pw_1/Conv2DЫ
pw_1/BiasAdd/ReadVariableOpReadVariableOp$pw_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pw_1/BiasAdd/ReadVariableOpЭ
pw_1/BiasAddBiasAddpw_1/Conv2D:output:0#pw_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
pw_1/BiasAdd╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1▐
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3pw_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Х
activation_2/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2
activation_2/Reluн
dw_2/depthwise/ReadVariableOpReadVariableOp&dw_2_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
dw_2/depthwise/ReadVariableOpЕ
dw_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dw_2/depthwise/ShapeН
dw_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dw_2/depthwise/dilation_rateф
dw_2/depthwiseDepthwiseConv2dNativeactivation_2/Relu:activations:0%dw_2/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
dw_2/depthwiseЫ
dw_2/BiasAdd/ReadVariableOpReadVariableOp$dw_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dw_2/BiasAdd/ReadVariableOpа
dw_2/BiasAddBiasAdddw_2/depthwise:output:0#dw_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
dw_2/BiasAdd╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1▐
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3dw_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3Х
activation_3/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2
activation_3/Reluд
pw_2/Conv2D/ReadVariableOpReadVariableOp#pw_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
pw_2/Conv2D/ReadVariableOp╠
pw_2/Conv2DConv2Dactivation_3/Relu:activations:0"pw_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
pw_2/Conv2DЫ
pw_2/BiasAdd/ReadVariableOpReadVariableOp$pw_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pw_2/BiasAdd/ReadVariableOpЭ
pw_2/BiasAddBiasAddpw_2/Conv2D:output:0#pw_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
pw_2/BiasAdd╢
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1▐
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3pw_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Х
activation_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2
activation_4/Relu╢
 dw_ee_1/depthwise/ReadVariableOpReadVariableOp)dw_ee_1_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02"
 dw_ee_1/depthwise/ReadVariableOpЛ
dw_ee_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dw_ee_1/depthwise/ShapeУ
dw_ee_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2!
dw_ee_1/depthwise/dilation_rateэ
dw_ee_1/depthwiseDepthwiseConv2dNativeactivation_4/Relu:activations:0(dw_ee_1/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Э*
paddingSAME*
strides
2
dw_ee_1/depthwiseд
dw_ee_1/BiasAdd/ReadVariableOpReadVariableOp'dw_ee_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dw_ee_1/BiasAdd/ReadVariableOpм
dw_ee_1/BiasAddBiasAdddw_ee_1/depthwise:output:0&dw_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Э2
dw_ee_1/BiasAddy
dw_ee_1/ReluReludw_ee_1/BiasAdd:output:0*
T0*0
_output_shapes
:         Э2
dw_ee_1/Relu╙
average_pooling2d/AvgPoolAvgPooldw_ee_1/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨   2
flatten/ConstЬ
flatten/ReshapeReshape"average_pooling2d/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ╨2
flatten/Reshape╡
"dense_1_ee_1/MatMul/ReadVariableOpReadVariableOp+dense_1_ee_1_matmul_readvariableop_resource*
_output_shapes
:	╨
*
dtype02$
"dense_1_ee_1/MatMul/ReadVariableOpм
dense_1_ee_1/MatMulMatMulflatten/Reshape:output:0*dense_1_ee_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_1_ee_1/MatMul│
#dense_1_ee_1/BiasAdd/ReadVariableOpReadVariableOp,dense_1_ee_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#dense_1_ee_1/BiasAdd/ReadVariableOp╡
dense_1_ee_1/BiasAddBiasAdddense_1_ee_1/MatMul:product:0+dense_1_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_1_ee_1/BiasAddИ
dense_1_ee_1/SoftmaxSoftmaxdense_1_ee_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_1_ee_1/Softmax┤
"dense_2_ee_1/MatMul/ReadVariableOpReadVariableOp+dense_2_ee_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"dense_2_ee_1/MatMul/ReadVariableOp▓
dense_2_ee_1/MatMulMatMuldense_1_ee_1/Softmax:softmax:0*dense_2_ee_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2_ee_1/MatMul│
#dense_2_ee_1/BiasAdd/ReadVariableOpReadVariableOp,dense_2_ee_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#dense_2_ee_1/BiasAdd/ReadVariableOp╡
dense_2_ee_1/BiasAddBiasAdddense_2_ee_1/MatMul:product:0+dense_2_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2_ee_1/BiasAddИ
dense_2_ee_1/SoftmaxSoftmaxdense_2_ee_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2_ee_1/Softmax┬
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul─
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp#pw_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpй
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_1/SquareС
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constв
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_1/mul/xд
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mul─
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp#pw_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpй
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_2/SquareС
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constв
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_2/mul/xд
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulr
IdentityIdentitydense_2_ee_1/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	:::::::::::::::::::::::::::::::::::::X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
┘
и
5__inference_batch_normalization_2_layer_call_fn_15001

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_130262
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
д
к
?__inference_dw_2_layer_call_and_return_conditional_losses_12369

inputs%
!depthwise_readvariableop_resource#
biasadd_readvariableop_resource
identityИЮ
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate═
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
	depthwiseМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЭ
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┘
и
5__inference_batch_normalization_1_layer_call_fn_14799

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_129352
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╙
Й
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12935

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣:::::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Ў
y
$__inference_pw_1_layer_call_fn_14735

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_1_layer_call_and_return_conditional_losses_128822
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╣::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
┬
°
?__inference_bn_1_layer_call_and_return_conditional_losses_12696

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ё:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ё:::::X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
Ю
н
B__inference_dw_ee_1_layer_call_and_return_conditional_losses_12601

inputs%
!depthwise_readvariableop_resource#
biasadd_readvariableop_resource
identityИЮ
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate═
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
	depthwiseМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЭ
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╝
H
,__inference_activation_2_layer_call_fn_14873

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_129762
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
║
y
$__inference_dw_2_layer_call_fn_12379

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_2_layer_call_and_return_conditional_losses_123692
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╛
`
'__inference_dropout_layer_call_fn_14561

inputs
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127572
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ё22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
┘
c
G__inference_activation_4_layer_call_and_return_conditional_losses_13185

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╣2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Ы
и
5__inference_batch_normalization_2_layer_call_fn_14924

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_124412
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12472

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12576

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ФN
Т
__inference__traced_save_15395
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop)
%savev2_bn_1_gamma_read_readvariableop(
$savev2_bn_1_beta_read_readvariableop/
+savev2_bn_1_moving_mean_read_readvariableop3
/savev2_bn_1_moving_variance_read_readvariableop4
0savev2_dw_1_depthwise_kernel_read_readvariableop(
$savev2_dw_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop*
&savev2_pw_1_kernel_read_readvariableop(
$savev2_pw_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop4
0savev2_dw_2_depthwise_kernel_read_readvariableop(
$savev2_dw_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop*
&savev2_pw_2_kernel_read_readvariableop(
$savev2_pw_2_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop7
3savev2_dw_ee_1_depthwise_kernel_read_readvariableop+
'savev2_dw_ee_1_bias_read_readvariableop2
.savev2_dense_1_ee_1_kernel_read_readvariableop0
,savev2_dense_1_ee_1_bias_read_readvariableop2
.savev2_dense_2_ee_1_kernel_read_readvariableop0
,savev2_dense_2_ee_1_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_aee35c43f30e41a099688836a4441b8c/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╠
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*▐
value╘B╤%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╥
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesї
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop%savev2_bn_1_gamma_read_readvariableop$savev2_bn_1_beta_read_readvariableop+savev2_bn_1_moving_mean_read_readvariableop/savev2_bn_1_moving_variance_read_readvariableop0savev2_dw_1_depthwise_kernel_read_readvariableop$savev2_dw_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop&savev2_pw_1_kernel_read_readvariableop$savev2_pw_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop0savev2_dw_2_depthwise_kernel_read_readvariableop$savev2_dw_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop&savev2_pw_2_kernel_read_readvariableop$savev2_pw_2_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop3savev2_dw_ee_1_depthwise_kernel_read_readvariableop'savev2_dw_ee_1_bias_read_readvariableop.savev2_dense_1_ee_1_kernel_read_readvariableop,savev2_dense_1_ee_1_bias_read_readvariableop.savev2_dense_2_ee_1_kernel_read_readvariableop,savev2_dense_2_ee_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*┬
_input_shapes░
н: :
::::::::::::::::::::::::::::::::	╨
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
::%!!

_output_shapes
:	╨
: "

_output_shapes
:
:$# 

_output_shapes

:
: $

_output_shapes
::%

_output_shapes
: 
Д
н
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13126

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Д{
х
G__inference_functional_1_layer_call_and_return_conditional_losses_13286
input_1
conv_1_12654
conv_1_12656

bn_1_12723

bn_1_12725

bn_1_12727

bn_1_12729

dw_1_12775

dw_1_12777
batch_normalization_12844
batch_normalization_12846
batch_normalization_12848
batch_normalization_12850

pw_1_12893

pw_1_12895
batch_normalization_1_12962
batch_normalization_1_12964
batch_normalization_1_12966
batch_normalization_1_12968

dw_2_12984

dw_2_12986
batch_normalization_2_13053
batch_normalization_2_13055
batch_normalization_2_13057
batch_normalization_2_13059

pw_2_13102

pw_2_13104
batch_normalization_3_13171
batch_normalization_3_13173
batch_normalization_3_13175
batch_normalization_3_13177
dw_ee_1_13193
dw_ee_1_13195
dense_1_ee_1_13235
dense_1_ee_1_13237
dense_2_ee_1_13262
dense_2_ee_1_13264
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвbn_1/StatefulPartitionedCallвconv_1/StatefulPartitionedCallв$dense_1_ee_1/StatefulPartitionedCallв$dense_2_ee_1/StatefulPartitionedCallвdropout/StatefulPartitionedCallвdw_1/StatefulPartitionedCallвdw_2/StatefulPartitionedCallвdw_ee_1/StatefulPartitionedCallвpw_1/StatefulPartitionedCallвpw_2/StatefulPartitionedCallС
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_1_12654conv_1_12656*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_126432 
conv_1/StatefulPartitionedCall┴
bn_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0
bn_1_12723
bn_1_12725
bn_1_12727
bn_1_12729*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_126782
bn_1/StatefulPartitionedCallБ
activation/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_127372
activation/PartitionedCallО
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127572!
dropout/StatefulPartitionedCallи
dw_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dw_1_12775
dw_1_12777*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_1_layer_call_and_return_conditional_losses_121382
dw_1/StatefulPartitionedCallи
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%dw_1/StatefulPartitionedCall:output:0batch_normalization_12844batch_normalization_12846batch_normalization_12848batch_normalization_12850*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_127992-
+batch_normalization/StatefulPartitionedCallЦ
activation_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_128582
activation_1/PartitionedCallе
pw_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0
pw_1_12893
pw_1_12895*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_1_layer_call_and_return_conditional_losses_128822
pw_1/StatefulPartitionedCall╢
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%pw_1/StatefulPartitionedCall:output:0batch_normalization_1_12962batch_normalization_1_12964batch_normalization_1_12966batch_normalization_1_12968*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_129172/
-batch_normalization_1/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_129762
activation_2/PartitionedCallе
dw_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0
dw_2_12984
dw_2_12986*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_2_layer_call_and_return_conditional_losses_123692
dw_2/StatefulPartitionedCall╢
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%dw_2/StatefulPartitionedCall:output:0batch_normalization_2_13053batch_normalization_2_13055batch_normalization_2_13057batch_normalization_2_13059*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_130082/
-batch_normalization_2/StatefulPartitionedCallШ
activation_3/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_130672
activation_3/PartitionedCallе
pw_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0
pw_2_13102
pw_2_13104*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_2_layer_call_and_return_conditional_losses_130912
pw_2/StatefulPartitionedCall╢
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%pw_2/StatefulPartitionedCall:output:0batch_normalization_3_13171batch_normalization_3_13173batch_normalization_3_13175batch_normalization_3_13177*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_131262/
-batch_normalization_3/StatefulPartitionedCallШ
activation_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_131852
activation_4/PartitionedCall┤
dw_ee_1/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dw_ee_1_13193dw_ee_1_13195*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dw_ee_1_layer_call_and_return_conditional_losses_126012!
dw_ee_1/StatefulPartitionedCallШ
!average_pooling2d/PartitionedCallPartitionedCall(dw_ee_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_126172#
!average_pooling2d/PartitionedCallї
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_132052
flatten/PartitionedCall┐
$dense_1_ee_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_ee_1_13235dense_1_ee_1_13237*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_132242&
$dense_1_ee_1/StatefulPartitionedCall╠
$dense_2_ee_1/StatefulPartitionedCallStatefulPartitionedCall-dense_1_ee_1/StatefulPartitionedCall:output:0dense_2_ee_1_13262dense_2_ee_1_13264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_132512&
$dense_2_ee_1/StatefulPartitionedCallй
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_1_12654*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulл
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp
pw_1_12893*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpй
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_1/SquareС
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constв
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_1/mul/xд
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulл
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp
pw_2_13102*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpй
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_2/SquareС
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constв
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_2/mul/xд
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulН
IdentityIdentity-dense_2_ee_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^conv_1/StatefulPartitionedCall%^dense_1_ee_1/StatefulPartitionedCall%^dense_2_ee_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^dw_1/StatefulPartitionedCall^dw_2/StatefulPartitionedCall ^dw_ee_1/StatefulPartitionedCall^pw_1/StatefulPartitionedCall^pw_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2L
$dense_1_ee_1/StatefulPartitionedCall$dense_1_ee_1/StatefulPartitionedCall2L
$dense_2_ee_1/StatefulPartitionedCall$dense_2_ee_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2<
dw_1/StatefulPartitionedCalldw_1/StatefulPartitionedCall2<
dw_2/StatefulPartitionedCalldw_2/StatefulPartitionedCall2B
dw_ee_1/StatefulPartitionedCalldw_ee_1/StatefulPartitionedCall2<
pw_1/StatefulPartitionedCallpw_1/StatefulPartitionedCall2<
pw_2/StatefulPartitionedCallpw_2/StatefulPartitionedCall:Y U
0
_output_shapes
:         т	
!
_user_specified_name	input_1
╚
a
B__inference_dropout_layer_call_and_return_conditional_losses_12757

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         ё2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         ё*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╟
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ё2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ё2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ё2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ё:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
·
{
&__inference_conv_1_layer_call_fn_14401

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_126432
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         т	::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
Д
н
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12917

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
є
Ь
?__inference_bn_1_layer_call_and_return_conditional_losses_14421

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ё:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ё::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
╔╬
Й
G__inference_functional_1_layer_call_and_return_conditional_losses_14058

inputs)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource 
bn_1_readvariableop_resource"
bn_1_readvariableop_1_resource1
-bn_1_fusedbatchnormv3_readvariableop_resource3
/bn_1_fusedbatchnormv3_readvariableop_1_resource*
&dw_1_depthwise_readvariableop_resource(
$dw_1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource'
#pw_1_conv2d_readvariableop_resource(
$pw_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
&dw_2_depthwise_readvariableop_resource(
$dw_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource'
#pw_2_conv2d_readvariableop_resource(
$pw_2_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource-
)dw_ee_1_depthwise_readvariableop_resource+
'dw_ee_1_biasadd_readvariableop_resource/
+dense_1_ee_1_matmul_readvariableop_resource0
,dense_1_ee_1_biasadd_readvariableop_resource/
+dense_2_ee_1_matmul_readvariableop_resource0
,dense_2_ee_1_biasadd_readvariableop_resource
identityИв"batch_normalization/AssignNewValueв$batch_normalization/AssignNewValue_1в$batch_normalization_1/AssignNewValueв&batch_normalization_1/AssignNewValue_1в$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в$batch_normalization_3/AssignNewValueв&batch_normalization_3/AssignNewValue_1вbn_1/AssignNewValueвbn_1/AssignNewValue_1к
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOp╣
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё*
paddingSAME*
strides
2
conv_1/Conv2Dб
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOpе
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё2
conv_1/BiasAddГ
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
:*
dtype02
bn_1/ReadVariableOpЙ
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn_1/ReadVariableOp_1╢
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp╝
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1И
bn_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ё:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
bn_1/FusedBatchNormV3Э
bn_1/AssignNewValueAssignVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource"bn_1/FusedBatchNormV3:batch_mean:0%^bn_1/FusedBatchNormV3/ReadVariableOp*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn_1/AssignNewValueл
bn_1/AssignNewValue_1AssignVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource&bn_1/FusedBatchNormV3:batch_variance:0'^bn_1/FusedBatchNormV3/ReadVariableOp_1*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn_1/AssignNewValue_1А
activation/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ё2
activation/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/dropout/Constл
dropout/dropout/MulMulactivation/Relu:activations:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         ё2
dropout/dropout/Mul{
dropout/dropout/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape╒
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         ё*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2 
dropout/dropout/GreaterEqual/yч
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ё2
dropout/dropout/GreaterEqualа
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ё2
dropout/dropout/Castг
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         ё2
dropout/dropout/Mul_1н
dw_1/depthwise/ReadVariableOpReadVariableOp&dw_1_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
dw_1/depthwise/ReadVariableOpЕ
dw_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dw_1/depthwise/ShapeН
dw_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dw_1/depthwise/dilation_rate▐
dw_1/depthwiseDepthwiseConv2dNativedropout/dropout/Mul_1:z:0%dw_1/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
dw_1/depthwiseЫ
dw_1/BiasAdd/ReadVariableOpReadVariableOp$dw_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dw_1/BiasAdd/ReadVariableOpа
dw_1/BiasAddBiasAdddw_1/depthwise:output:0#dw_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
dw_1/BiasAdd░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1р
$batch_normalization/FusedBatchNormV3FusedBatchNormV3dw_1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2&
$batch_normalization/FusedBatchNormV3ў
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValueЕ
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1У
activation_1/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2
activation_1/Reluд
pw_1/Conv2D/ReadVariableOpReadVariableOp#pw_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
pw_1/Conv2D/ReadVariableOp╠
pw_1/Conv2DConv2Dactivation_1/Relu:activations:0"pw_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
pw_1/Conv2DЫ
pw_1/BiasAdd/ReadVariableOpReadVariableOp$pw_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pw_1/BiasAdd/ReadVariableOpЭ
pw_1/BiasAddBiasAddpw_1/Conv2D:output:0#pw_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
pw_1/BiasAdd╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ь
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3pw_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_1/FusedBatchNormV3Г
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueС
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1Х
activation_2/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2
activation_2/Reluн
dw_2/depthwise/ReadVariableOpReadVariableOp&dw_2_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
dw_2/depthwise/ReadVariableOpЕ
dw_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dw_2/depthwise/ShapeН
dw_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dw_2/depthwise/dilation_rateф
dw_2/depthwiseDepthwiseConv2dNativeactivation_2/Relu:activations:0%dw_2/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
dw_2/depthwiseЫ
dw_2/BiasAdd/ReadVariableOpReadVariableOp$dw_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dw_2/BiasAdd/ReadVariableOpа
dw_2/BiasAddBiasAdddw_2/depthwise:output:0#dw_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
dw_2/BiasAdd╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ь
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3dw_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_2/FusedBatchNormV3Г
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueС
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1Х
activation_3/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2
activation_3/Reluд
pw_2/Conv2D/ReadVariableOpReadVariableOp#pw_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
pw_2/Conv2D/ReadVariableOp╠
pw_2/Conv2DConv2Dactivation_3/Relu:activations:0"pw_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
pw_2/Conv2DЫ
pw_2/BiasAdd/ReadVariableOpReadVariableOp$pw_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pw_2/BiasAdd/ReadVariableOpЭ
pw_2/BiasAddBiasAddpw_2/Conv2D:output:0#pw_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
pw_2/BiasAdd╢
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1щ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ь
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3pw_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_3/FusedBatchNormV3Г
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValueС
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1Х
activation_4/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2
activation_4/Relu╢
 dw_ee_1/depthwise/ReadVariableOpReadVariableOp)dw_ee_1_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02"
 dw_ee_1/depthwise/ReadVariableOpЛ
dw_ee_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
dw_ee_1/depthwise/ShapeУ
dw_ee_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2!
dw_ee_1/depthwise/dilation_rateэ
dw_ee_1/depthwiseDepthwiseConv2dNativeactivation_4/Relu:activations:0(dw_ee_1/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Э*
paddingSAME*
strides
2
dw_ee_1/depthwiseд
dw_ee_1/BiasAdd/ReadVariableOpReadVariableOp'dw_ee_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dw_ee_1/BiasAdd/ReadVariableOpм
dw_ee_1/BiasAddBiasAdddw_ee_1/depthwise:output:0&dw_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Э2
dw_ee_1/BiasAddy
dw_ee_1/ReluReludw_ee_1/BiasAdd:output:0*
T0*0
_output_shapes
:         Э2
dw_ee_1/Relu╙
average_pooling2d/AvgPoolAvgPooldw_ee_1/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨   2
flatten/ConstЬ
flatten/ReshapeReshape"average_pooling2d/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         ╨2
flatten/Reshape╡
"dense_1_ee_1/MatMul/ReadVariableOpReadVariableOp+dense_1_ee_1_matmul_readvariableop_resource*
_output_shapes
:	╨
*
dtype02$
"dense_1_ee_1/MatMul/ReadVariableOpм
dense_1_ee_1/MatMulMatMulflatten/Reshape:output:0*dense_1_ee_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_1_ee_1/MatMul│
#dense_1_ee_1/BiasAdd/ReadVariableOpReadVariableOp,dense_1_ee_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#dense_1_ee_1/BiasAdd/ReadVariableOp╡
dense_1_ee_1/BiasAddBiasAdddense_1_ee_1/MatMul:product:0+dense_1_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_1_ee_1/BiasAddИ
dense_1_ee_1/SoftmaxSoftmaxdense_1_ee_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_1_ee_1/Softmax┤
"dense_2_ee_1/MatMul/ReadVariableOpReadVariableOp+dense_2_ee_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"dense_2_ee_1/MatMul/ReadVariableOp▓
dense_2_ee_1/MatMulMatMuldense_1_ee_1/Softmax:softmax:0*dense_2_ee_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2_ee_1/MatMul│
#dense_2_ee_1/BiasAdd/ReadVariableOpReadVariableOp,dense_2_ee_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#dense_2_ee_1/BiasAdd/ReadVariableOp╡
dense_2_ee_1/BiasAddBiasAdddense_2_ee_1/MatMul:product:0+dense_2_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2_ee_1/BiasAddИ
dense_2_ee_1/SoftmaxSoftmaxdense_2_ee_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2_ee_1/Softmax┬
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul─
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp#pw_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpй
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_1/SquareС
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constв
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_1/mul/xд
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mul─
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp#pw_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpй
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_2/SquareС
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constв
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_2/mul/xд
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mul▄
IdentityIdentitydense_2_ee_1/Softmax:softmax:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1^bn_1/AssignNewValue^bn_1/AssignNewValue_1*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12*
bn_1/AssignNewValuebn_1/AssignNewValue2.
bn_1/AssignNewValue_1bn_1/AssignNewValue_1:X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
Ы
и
5__inference_batch_normalization_1_layer_call_fn_14850

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_123142
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12545

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Э
и
5__inference_batch_normalization_1_layer_call_fn_14863

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_123452
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┘
c
G__inference_activation_3_layer_call_and_return_conditional_losses_13067

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╣2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
┘
c
G__inference_activation_1_layer_call_and_return_conditional_losses_14699

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╣2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Ы
и
5__inference_batch_normalization_3_layer_call_fn_15093

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_125452
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╫
a
E__inference_activation_layer_call_and_return_conditional_losses_14534

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ё2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ё:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
цy
├
G__inference_functional_1_layer_call_and_return_conditional_losses_13401
input_1
conv_1_13289
conv_1_13291

bn_1_13294

bn_1_13296

bn_1_13298

bn_1_13300

dw_1_13305

dw_1_13307
batch_normalization_13310
batch_normalization_13312
batch_normalization_13314
batch_normalization_13316

pw_1_13320

pw_1_13322
batch_normalization_1_13325
batch_normalization_1_13327
batch_normalization_1_13329
batch_normalization_1_13331

dw_2_13335

dw_2_13337
batch_normalization_2_13340
batch_normalization_2_13342
batch_normalization_2_13344
batch_normalization_2_13346

pw_2_13350

pw_2_13352
batch_normalization_3_13355
batch_normalization_3_13357
batch_normalization_3_13359
batch_normalization_3_13361
dw_ee_1_13365
dw_ee_1_13367
dense_1_ee_1_13372
dense_1_ee_1_13374
dense_2_ee_1_13377
dense_2_ee_1_13379
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвbn_1/StatefulPartitionedCallвconv_1/StatefulPartitionedCallв$dense_1_ee_1/StatefulPartitionedCallв$dense_2_ee_1/StatefulPartitionedCallвdw_1/StatefulPartitionedCallвdw_2/StatefulPartitionedCallвdw_ee_1/StatefulPartitionedCallвpw_1/StatefulPartitionedCallвpw_2/StatefulPartitionedCallС
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_1_13289conv_1_13291*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_126432 
conv_1/StatefulPartitionedCall├
bn_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0
bn_1_13294
bn_1_13296
bn_1_13298
bn_1_13300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_126962
bn_1/StatefulPartitionedCallБ
activation/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_127372
activation/PartitionedCallЎ
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127622
dropout/PartitionedCallа
dw_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dw_1_13305
dw_1_13307*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_1_layer_call_and_return_conditional_losses_121382
dw_1/StatefulPartitionedCallк
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%dw_1/StatefulPartitionedCall:output:0batch_normalization_13310batch_normalization_13312batch_normalization_13314batch_normalization_13316*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_128172-
+batch_normalization/StatefulPartitionedCallЦ
activation_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_128582
activation_1/PartitionedCallе
pw_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0
pw_1_13320
pw_1_13322*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_1_layer_call_and_return_conditional_losses_128822
pw_1/StatefulPartitionedCall╕
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%pw_1/StatefulPartitionedCall:output:0batch_normalization_1_13325batch_normalization_1_13327batch_normalization_1_13329batch_normalization_1_13331*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_129352/
-batch_normalization_1/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_129762
activation_2/PartitionedCallе
dw_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0
dw_2_13335
dw_2_13337*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_2_layer_call_and_return_conditional_losses_123692
dw_2/StatefulPartitionedCall╕
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%dw_2/StatefulPartitionedCall:output:0batch_normalization_2_13340batch_normalization_2_13342batch_normalization_2_13344batch_normalization_2_13346*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_130262/
-batch_normalization_2/StatefulPartitionedCallШ
activation_3/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_130672
activation_3/PartitionedCallе
pw_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0
pw_2_13350
pw_2_13352*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_2_layer_call_and_return_conditional_losses_130912
pw_2/StatefulPartitionedCall╕
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%pw_2/StatefulPartitionedCall:output:0batch_normalization_3_13355batch_normalization_3_13357batch_normalization_3_13359batch_normalization_3_13361*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_131442/
-batch_normalization_3/StatefulPartitionedCallШ
activation_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_131852
activation_4/PartitionedCall┤
dw_ee_1/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dw_ee_1_13365dw_ee_1_13367*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dw_ee_1_layer_call_and_return_conditional_losses_126012!
dw_ee_1/StatefulPartitionedCallШ
!average_pooling2d/PartitionedCallPartitionedCall(dw_ee_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_126172#
!average_pooling2d/PartitionedCallї
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_132052
flatten/PartitionedCall┐
$dense_1_ee_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_ee_1_13372dense_1_ee_1_13374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_132242&
$dense_1_ee_1/StatefulPartitionedCall╠
$dense_2_ee_1/StatefulPartitionedCallStatefulPartitionedCall-dense_1_ee_1/StatefulPartitionedCall:output:0dense_2_ee_1_13377dense_2_ee_1_13379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_132512&
$dense_2_ee_1/StatefulPartitionedCallй
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_1_13289*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulл
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp
pw_1_13320*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpй
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_1/SquareС
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constв
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_1/mul/xд
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulл
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp
pw_2_13350*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpй
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_2/SquareС
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constв
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_2/mul/xд
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulы
IdentityIdentity-dense_2_ee_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^conv_1/StatefulPartitionedCall%^dense_1_ee_1/StatefulPartitionedCall%^dense_2_ee_1/StatefulPartitionedCall^dw_1/StatefulPartitionedCall^dw_2/StatefulPartitionedCall ^dw_ee_1/StatefulPartitionedCall^pw_1/StatefulPartitionedCall^pw_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2L
$dense_1_ee_1/StatefulPartitionedCall$dense_1_ee_1/StatefulPartitionedCall2L
$dense_2_ee_1/StatefulPartitionedCall$dense_2_ee_1/StatefulPartitionedCall2<
dw_1/StatefulPartitionedCalldw_1/StatefulPartitionedCall2<
dw_2/StatefulPartitionedCalldw_2/StatefulPartitionedCall2B
dw_ee_1/StatefulPartitionedCalldw_ee_1/StatefulPartitionedCall2<
pw_1/StatefulPartitionedCallpw_1/StatefulPartitionedCall2<
pw_2/StatefulPartitionedCallpw_2/StatefulPartitionedCall:Y U
0
_output_shapes
:         т	
!
_user_specified_name	input_1
╙
Й
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_13144

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣:::::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╖
п
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_13224

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╨
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╨:::P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
В
л
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14650

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╕
F
*__inference_activation_layer_call_fn_14539

inputs
identity╠
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_127372
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ё:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
Ў
y
$__inference_pw_2_layer_call_fn_15042

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_2_layer_call_and_return_conditional_losses_130912
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╣::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
п
M
1__inference_average_pooling2d_layer_call_fn_12623

inputs
identityэ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_126172
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┬
°
?__inference_bn_1_layer_call_and_return_conditional_losses_14439

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ё:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ё:::::X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
┘
c
G__inference_activation_1_layer_call_and_return_conditional_losses_12858

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╣2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Б{
ф
G__inference_functional_1_layer_call_and_return_conditional_losses_13519

inputs
conv_1_13407
conv_1_13409

bn_1_13412

bn_1_13414

bn_1_13416

bn_1_13418

dw_1_13423

dw_1_13425
batch_normalization_13428
batch_normalization_13430
batch_normalization_13432
batch_normalization_13434

pw_1_13438

pw_1_13440
batch_normalization_1_13443
batch_normalization_1_13445
batch_normalization_1_13447
batch_normalization_1_13449

dw_2_13453

dw_2_13455
batch_normalization_2_13458
batch_normalization_2_13460
batch_normalization_2_13462
batch_normalization_2_13464

pw_2_13468

pw_2_13470
batch_normalization_3_13473
batch_normalization_3_13475
batch_normalization_3_13477
batch_normalization_3_13479
dw_ee_1_13483
dw_ee_1_13485
dense_1_ee_1_13490
dense_1_ee_1_13492
dense_2_ee_1_13495
dense_2_ee_1_13497
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвbn_1/StatefulPartitionedCallвconv_1/StatefulPartitionedCallв$dense_1_ee_1/StatefulPartitionedCallв$dense_2_ee_1/StatefulPartitionedCallвdropout/StatefulPartitionedCallвdw_1/StatefulPartitionedCallвdw_2/StatefulPartitionedCallвdw_ee_1/StatefulPartitionedCallвpw_1/StatefulPartitionedCallвpw_2/StatefulPartitionedCallР
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_13407conv_1_13409*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_126432 
conv_1/StatefulPartitionedCall┴
bn_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0
bn_1_13412
bn_1_13414
bn_1_13416
bn_1_13418*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_126782
bn_1/StatefulPartitionedCallБ
activation/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_127372
activation/PartitionedCallО
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127572!
dropout/StatefulPartitionedCallи
dw_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dw_1_13423
dw_1_13425*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_1_layer_call_and_return_conditional_losses_121382
dw_1/StatefulPartitionedCallи
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%dw_1/StatefulPartitionedCall:output:0batch_normalization_13428batch_normalization_13430batch_normalization_13432batch_normalization_13434*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_127992-
+batch_normalization/StatefulPartitionedCallЦ
activation_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_128582
activation_1/PartitionedCallе
pw_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0
pw_1_13438
pw_1_13440*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_1_layer_call_and_return_conditional_losses_128822
pw_1/StatefulPartitionedCall╢
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%pw_1/StatefulPartitionedCall:output:0batch_normalization_1_13443batch_normalization_1_13445batch_normalization_1_13447batch_normalization_1_13449*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_129172/
-batch_normalization_1/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_129762
activation_2/PartitionedCallе
dw_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0
dw_2_13453
dw_2_13455*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_2_layer_call_and_return_conditional_losses_123692
dw_2/StatefulPartitionedCall╢
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%dw_2/StatefulPartitionedCall:output:0batch_normalization_2_13458batch_normalization_2_13460batch_normalization_2_13462batch_normalization_2_13464*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_130082/
-batch_normalization_2/StatefulPartitionedCallШ
activation_3/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_130672
activation_3/PartitionedCallе
pw_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0
pw_2_13468
pw_2_13470*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_2_layer_call_and_return_conditional_losses_130912
pw_2/StatefulPartitionedCall╢
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%pw_2/StatefulPartitionedCall:output:0batch_normalization_3_13473batch_normalization_3_13475batch_normalization_3_13477batch_normalization_3_13479*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_131262/
-batch_normalization_3/StatefulPartitionedCallШ
activation_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_131852
activation_4/PartitionedCall┤
dw_ee_1/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dw_ee_1_13483dw_ee_1_13485*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dw_ee_1_layer_call_and_return_conditional_losses_126012!
dw_ee_1/StatefulPartitionedCallШ
!average_pooling2d/PartitionedCallPartitionedCall(dw_ee_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_126172#
!average_pooling2d/PartitionedCallї
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_132052
flatten/PartitionedCall┐
$dense_1_ee_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_ee_1_13490dense_1_ee_1_13492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_132242&
$dense_1_ee_1/StatefulPartitionedCall╠
$dense_2_ee_1/StatefulPartitionedCallStatefulPartitionedCall-dense_1_ee_1/StatefulPartitionedCall:output:0dense_2_ee_1_13495dense_2_ee_1_13497*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_132512&
$dense_2_ee_1/StatefulPartitionedCallй
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_1_13407*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulл
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp
pw_1_13438*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpй
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_1/SquareС
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constв
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_1/mul/xд
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulл
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp
pw_2_13468*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpй
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_2/SquareС
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constв
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_2/mul/xд
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulН
IdentityIdentity-dense_2_ee_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^conv_1/StatefulPartitionedCall%^dense_1_ee_1/StatefulPartitionedCall%^dense_2_ee_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^dw_1/StatefulPartitionedCall^dw_2/StatefulPartitionedCall ^dw_ee_1/StatefulPartitionedCall^pw_1/StatefulPartitionedCall^pw_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2L
$dense_1_ee_1/StatefulPartitionedCall$dense_1_ee_1/StatefulPartitionedCall2L
$dense_2_ee_1/StatefulPartitionedCall$dense_2_ee_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2<
dw_1/StatefulPartitionedCalldw_1/StatefulPartitionedCall2<
dw_2/StatefulPartitionedCalldw_2/StatefulPartitionedCall2B
dw_ee_1/StatefulPartitionedCalldw_ee_1/StatefulPartitionedCall2<
pw_1/StatefulPartitionedCallpw_1/StatefulPartitionedCall2<
pw_2/StatefulPartitionedCallpw_2/StatefulPartitionedCall:X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
щ
`
B__inference_dropout_layer_call_and_return_conditional_losses_12762

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         ё2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ё2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         ё:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
д
к
?__inference_dw_1_layer_call_and_return_conditional_losses_12138

inputs%
!depthwise_readvariableop_resource#
biasadd_readvariableop_resource
identityИЮ
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/ShapeГ
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rate═
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
	depthwiseМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЭ
BiasAddBiasAdddepthwise:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
уy
┬
G__inference_functional_1_layer_call_and_return_conditional_losses_13711

inputs
conv_1_13599
conv_1_13601

bn_1_13604

bn_1_13606

bn_1_13608

bn_1_13610

dw_1_13615

dw_1_13617
batch_normalization_13620
batch_normalization_13622
batch_normalization_13624
batch_normalization_13626

pw_1_13630

pw_1_13632
batch_normalization_1_13635
batch_normalization_1_13637
batch_normalization_1_13639
batch_normalization_1_13641

dw_2_13645

dw_2_13647
batch_normalization_2_13650
batch_normalization_2_13652
batch_normalization_2_13654
batch_normalization_2_13656

pw_2_13660

pw_2_13662
batch_normalization_3_13665
batch_normalization_3_13667
batch_normalization_3_13669
batch_normalization_3_13671
dw_ee_1_13675
dw_ee_1_13677
dense_1_ee_1_13682
dense_1_ee_1_13684
dense_2_ee_1_13687
dense_2_ee_1_13689
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallвbn_1/StatefulPartitionedCallвconv_1/StatefulPartitionedCallв$dense_1_ee_1/StatefulPartitionedCallв$dense_2_ee_1/StatefulPartitionedCallвdw_1/StatefulPartitionedCallвdw_2/StatefulPartitionedCallвdw_ee_1/StatefulPartitionedCallвpw_1/StatefulPartitionedCallвpw_2/StatefulPartitionedCallР
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_13599conv_1_13601*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_126432 
conv_1/StatefulPartitionedCall├
bn_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0
bn_1_13604
bn_1_13606
bn_1_13608
bn_1_13610*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_126962
bn_1/StatefulPartitionedCallБ
activation/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_127372
activation/PartitionedCallЎ
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127622
dropout/PartitionedCallа
dw_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dw_1_13615
dw_1_13617*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_1_layer_call_and_return_conditional_losses_121382
dw_1/StatefulPartitionedCallк
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%dw_1/StatefulPartitionedCall:output:0batch_normalization_13620batch_normalization_13622batch_normalization_13624batch_normalization_13626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_128172-
+batch_normalization/StatefulPartitionedCallЦ
activation_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_128582
activation_1/PartitionedCallе
pw_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0
pw_1_13630
pw_1_13632*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_1_layer_call_and_return_conditional_losses_128822
pw_1/StatefulPartitionedCall╕
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%pw_1/StatefulPartitionedCall:output:0batch_normalization_1_13635batch_normalization_1_13637batch_normalization_1_13639batch_normalization_1_13641*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_129352/
-batch_normalization_1/StatefulPartitionedCallШ
activation_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_129762
activation_2/PartitionedCallе
dw_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0
dw_2_13645
dw_2_13647*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_2_layer_call_and_return_conditional_losses_123692
dw_2/StatefulPartitionedCall╕
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%dw_2/StatefulPartitionedCall:output:0batch_normalization_2_13650batch_normalization_2_13652batch_normalization_2_13654batch_normalization_2_13656*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_130262/
-batch_normalization_2/StatefulPartitionedCallШ
activation_3/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_130672
activation_3/PartitionedCallе
pw_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0
pw_2_13660
pw_2_13662*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_pw_2_layer_call_and_return_conditional_losses_130912
pw_2/StatefulPartitionedCall╕
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%pw_2/StatefulPartitionedCall:output:0batch_normalization_3_13665batch_normalization_3_13667batch_normalization_3_13669batch_normalization_3_13671*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_131442/
-batch_normalization_3/StatefulPartitionedCallШ
activation_4/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_131852
activation_4/PartitionedCall┤
dw_ee_1/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0dw_ee_1_13675dw_ee_1_13677*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         Э*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dw_ee_1_layer_call_and_return_conditional_losses_126012!
dw_ee_1/StatefulPartitionedCallШ
!average_pooling2d/PartitionedCallPartitionedCall(dw_ee_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_126172#
!average_pooling2d/PartitionedCallї
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_132052
flatten/PartitionedCall┐
$dense_1_ee_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_ee_1_13682dense_1_ee_1_13684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_132242&
$dense_1_ee_1/StatefulPartitionedCall╠
$dense_2_ee_1/StatefulPartitionedCallStatefulPartitionedCall-dense_1_ee_1/StatefulPartitionedCall:output:0dense_2_ee_1_13687dense_2_ee_1_13689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_132512&
$dense_2_ee_1/StatefulPartitionedCallй
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_1_13599*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulл
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp
pw_1_13630*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_1/Square/ReadVariableOpй
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_1/SquareС
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_1/Constв
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/Sum}
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_1/mul/xд
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_1/mulл
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp
pw_2_13660*&
_output_shapes
:*
dtype02,
*kernel/Regularizer_2/Square/ReadVariableOpй
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer_2/SquareС
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer_2/Constв
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/Sum}
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer_2/mul/xд
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer_2/mulы
IdentityIdentity-dense_2_ee_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^conv_1/StatefulPartitionedCall%^dense_1_ee_1/StatefulPartitionedCall%^dense_2_ee_1/StatefulPartitionedCall^dw_1/StatefulPartitionedCall^dw_2/StatefulPartitionedCall ^dw_ee_1/StatefulPartitionedCall^pw_1/StatefulPartitionedCall^pw_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2L
$dense_1_ee_1/StatefulPartitionedCall$dense_1_ee_1/StatefulPartitionedCall2L
$dense_2_ee_1/StatefulPartitionedCall$dense_2_ee_1/StatefulPartitionedCall2<
dw_1/StatefulPartitionedCalldw_1/StatefulPartitionedCall2<
dw_2/StatefulPartitionedCalldw_2/StatefulPartitionedCall2B
dw_ee_1/StatefulPartitionedCalldw_ee_1/StatefulPartitionedCall2<
pw_1/StatefulPartitionedCallpw_1/StatefulPartitionedCall2<
pw_2/StatefulPartitionedCallpw_2/StatefulPartitionedCall:X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
╨
з
?__inference_pw_1_layer_call_and_return_conditional_losses_14726

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2	
BiasAdd╗
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╣:::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12345

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╥
й
A__inference_conv_1_layer_call_and_return_conditional_losses_14392

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё2	
BiasAdd╗
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         т	:::X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
┘
c
G__inference_activation_2_layer_call_and_return_conditional_losses_14868

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╣2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
▓
C
'__inference_dropout_layer_call_fn_14566

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_127622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ё:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14911

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Э
и
5__inference_batch_normalization_2_layer_call_fn_14937

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_124722
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
√
Ч
$__inference_bn_1_layer_call_fn_14529

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_121142
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ц
З
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14604

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╫
и
5__inference_batch_normalization_1_layer_call_fn_14786

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_129172
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14837

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
хв
═
 __inference__wrapped_model_12021
input_16
2functional_1_conv_1_conv2d_readvariableop_resource7
3functional_1_conv_1_biasadd_readvariableop_resource-
)functional_1_bn_1_readvariableop_resource/
+functional_1_bn_1_readvariableop_1_resource>
:functional_1_bn_1_fusedbatchnormv3_readvariableop_resource@
<functional_1_bn_1_fusedbatchnormv3_readvariableop_1_resource7
3functional_1_dw_1_depthwise_readvariableop_resource5
1functional_1_dw_1_biasadd_readvariableop_resource<
8functional_1_batch_normalization_readvariableop_resource>
:functional_1_batch_normalization_readvariableop_1_resourceM
Ifunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_resourceO
Kfunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_resource4
0functional_1_pw_1_conv2d_readvariableop_resource5
1functional_1_pw_1_biasadd_readvariableop_resource>
:functional_1_batch_normalization_1_readvariableop_resource@
<functional_1_batch_normalization_1_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3functional_1_dw_2_depthwise_readvariableop_resource5
1functional_1_dw_2_biasadd_readvariableop_resource>
:functional_1_batch_normalization_2_readvariableop_resource@
<functional_1_batch_normalization_2_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource4
0functional_1_pw_2_conv2d_readvariableop_resource5
1functional_1_pw_2_biasadd_readvariableop_resource>
:functional_1_batch_normalization_3_readvariableop_resource@
<functional_1_batch_normalization_3_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:
6functional_1_dw_ee_1_depthwise_readvariableop_resource8
4functional_1_dw_ee_1_biasadd_readvariableop_resource<
8functional_1_dense_1_ee_1_matmul_readvariableop_resource=
9functional_1_dense_1_ee_1_biasadd_readvariableop_resource<
8functional_1_dense_2_ee_1_matmul_readvariableop_resource=
9functional_1_dense_2_ee_1_biasadd_readvariableop_resource
identityИ╤
)functional_1/conv_1/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02+
)functional_1/conv_1/Conv2D/ReadVariableOpс
functional_1/conv_1/Conv2DConv2Dinput_11functional_1/conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё*
paddingSAME*
strides
2
functional_1/conv_1/Conv2D╚
*functional_1/conv_1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/conv_1/BiasAdd/ReadVariableOp┘
functional_1/conv_1/BiasAddBiasAdd#functional_1/conv_1/Conv2D:output:02functional_1/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё2
functional_1/conv_1/BiasAddк
 functional_1/bn_1/ReadVariableOpReadVariableOp)functional_1_bn_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 functional_1/bn_1/ReadVariableOp░
"functional_1/bn_1/ReadVariableOp_1ReadVariableOp+functional_1_bn_1_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"functional_1/bn_1/ReadVariableOp_1▌
1functional_1/bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp:functional_1_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_1/bn_1/FusedBatchNormV3/ReadVariableOpу
3functional_1/bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<functional_1_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3functional_1/bn_1/FusedBatchNormV3/ReadVariableOp_1╒
"functional_1/bn_1/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_1/BiasAdd:output:0(functional_1/bn_1/ReadVariableOp:value:0*functional_1/bn_1/ReadVariableOp_1:value:09functional_1/bn_1/FusedBatchNormV3/ReadVariableOp:value:0;functional_1/bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ё:::::*
epsilon%oГ:*
is_training( 2$
"functional_1/bn_1/FusedBatchNormV3з
functional_1/activation/ReluRelu&functional_1/bn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ё2
functional_1/activation/Relu▒
functional_1/dropout/IdentityIdentity*functional_1/activation/Relu:activations:0*
T0*0
_output_shapes
:         ё2
functional_1/dropout/Identity╘
*functional_1/dw_1/depthwise/ReadVariableOpReadVariableOp3functional_1_dw_1_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02,
*functional_1/dw_1/depthwise/ReadVariableOpЯ
!functional_1/dw_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2#
!functional_1/dw_1/depthwise/Shapeз
)functional_1/dw_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2+
)functional_1/dw_1/depthwise/dilation_rateТ
functional_1/dw_1/depthwiseDepthwiseConv2dNative&functional_1/dropout/Identity:output:02functional_1/dw_1/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
functional_1/dw_1/depthwise┬
(functional_1/dw_1/BiasAdd/ReadVariableOpReadVariableOp1functional_1_dw_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(functional_1/dw_1/BiasAdd/ReadVariableOp╘
functional_1/dw_1/BiasAddBiasAdd$functional_1/dw_1/depthwise:output:00functional_1/dw_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
functional_1/dw_1/BiasAdd╫
/functional_1/batch_normalization/ReadVariableOpReadVariableOp8functional_1_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype021
/functional_1/batch_normalization/ReadVariableOp▌
1functional_1/batch_normalization/ReadVariableOp_1ReadVariableOp:functional_1_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype023
1functional_1/batch_normalization/ReadVariableOp_1К
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpIfunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOpР
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKfunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1н
1functional_1/batch_normalization/FusedBatchNormV3FusedBatchNormV3"functional_1/dw_1/BiasAdd:output:07functional_1/batch_normalization/ReadVariableOp:value:09functional_1/batch_normalization/ReadVariableOp_1:value:0Hfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 23
1functional_1/batch_normalization/FusedBatchNormV3║
functional_1/activation_1/ReluRelu5functional_1/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2 
functional_1/activation_1/Relu╦
'functional_1/pw_1/Conv2D/ReadVariableOpReadVariableOp0functional_1_pw_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'functional_1/pw_1/Conv2D/ReadVariableOpА
functional_1/pw_1/Conv2DConv2D,functional_1/activation_1/Relu:activations:0/functional_1/pw_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
functional_1/pw_1/Conv2D┬
(functional_1/pw_1/BiasAdd/ReadVariableOpReadVariableOp1functional_1_pw_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(functional_1/pw_1/BiasAdd/ReadVariableOp╤
functional_1/pw_1/BiasAddBiasAdd!functional_1/pw_1/Conv2D:output:00functional_1/pw_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
functional_1/pw_1/BiasAdd▌
1functional_1/batch_normalization_1/ReadVariableOpReadVariableOp:functional_1_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_1/batch_normalization_1/ReadVariableOpу
3functional_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype025
3functional_1/batch_normalization_1/ReadVariableOp_1Р
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЦ
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1╣
3functional_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3"functional_1/pw_1/BiasAdd:output:09functional_1/batch_normalization_1/ReadVariableOp:value:0;functional_1/batch_normalization_1/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_1/FusedBatchNormV3╝
functional_1/activation_2/ReluRelu7functional_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2 
functional_1/activation_2/Relu╘
*functional_1/dw_2/depthwise/ReadVariableOpReadVariableOp3functional_1_dw_2_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02,
*functional_1/dw_2/depthwise/ReadVariableOpЯ
!functional_1/dw_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2#
!functional_1/dw_2/depthwise/Shapeз
)functional_1/dw_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2+
)functional_1/dw_2/depthwise/dilation_rateШ
functional_1/dw_2/depthwiseDepthwiseConv2dNative,functional_1/activation_2/Relu:activations:02functional_1/dw_2/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
functional_1/dw_2/depthwise┬
(functional_1/dw_2/BiasAdd/ReadVariableOpReadVariableOp1functional_1_dw_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(functional_1/dw_2/BiasAdd/ReadVariableOp╘
functional_1/dw_2/BiasAddBiasAdd$functional_1/dw_2/depthwise:output:00functional_1/dw_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
functional_1/dw_2/BiasAdd▌
1functional_1/batch_normalization_2/ReadVariableOpReadVariableOp:functional_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_1/batch_normalization_2/ReadVariableOpу
3functional_1/batch_normalization_2/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype025
3functional_1/batch_normalization_2/ReadVariableOp_1Р
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЦ
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1╣
3functional_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3"functional_1/dw_2/BiasAdd:output:09functional_1/batch_normalization_2/ReadVariableOp:value:0;functional_1/batch_normalization_2/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_2/FusedBatchNormV3╝
functional_1/activation_3/ReluRelu7functional_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2 
functional_1/activation_3/Relu╦
'functional_1/pw_2/Conv2D/ReadVariableOpReadVariableOp0functional_1_pw_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'functional_1/pw_2/Conv2D/ReadVariableOpА
functional_1/pw_2/Conv2DConv2D,functional_1/activation_3/Relu:activations:0/functional_1/pw_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
functional_1/pw_2/Conv2D┬
(functional_1/pw_2/BiasAdd/ReadVariableOpReadVariableOp1functional_1_pw_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(functional_1/pw_2/BiasAdd/ReadVariableOp╤
functional_1/pw_2/BiasAddBiasAdd!functional_1/pw_2/Conv2D:output:00functional_1/pw_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2
functional_1/pw_2/BiasAdd▌
1functional_1/batch_normalization_3/ReadVariableOpReadVariableOp:functional_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_1/batch_normalization_3/ReadVariableOpу
3functional_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3functional_1/batch_normalization_3/ReadVariableOp_1Р
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЦ
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1╣
3functional_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3"functional_1/pw_2/BiasAdd:output:09functional_1/batch_normalization_3/ReadVariableOp:value:0;functional_1/batch_normalization_3/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 25
3functional_1/batch_normalization_3/FusedBatchNormV3╝
functional_1/activation_4/ReluRelu7functional_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2 
functional_1/activation_4/Relu▌
-functional_1/dw_ee_1/depthwise/ReadVariableOpReadVariableOp6functional_1_dw_ee_1_depthwise_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_1/dw_ee_1/depthwise/ReadVariableOpе
$functional_1/dw_ee_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2&
$functional_1/dw_ee_1/depthwise/Shapeн
,functional_1/dw_ee_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2.
,functional_1/dw_ee_1/depthwise/dilation_rateб
functional_1/dw_ee_1/depthwiseDepthwiseConv2dNative,functional_1/activation_4/Relu:activations:05functional_1/dw_ee_1/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Э*
paddingSAME*
strides
2 
functional_1/dw_ee_1/depthwise╦
+functional_1/dw_ee_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dw_ee_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dw_ee_1/BiasAdd/ReadVariableOpр
functional_1/dw_ee_1/BiasAddBiasAdd'functional_1/dw_ee_1/depthwise:output:03functional_1/dw_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Э2
functional_1/dw_ee_1/BiasAddа
functional_1/dw_ee_1/ReluRelu%functional_1/dw_ee_1/BiasAdd:output:0*
T0*0
_output_shapes
:         Э2
functional_1/dw_ee_1/Relu·
&functional_1/average_pooling2d/AvgPoolAvgPool'functional_1/dw_ee_1/Relu:activations:0*
T0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2(
&functional_1/average_pooling2d/AvgPoolЙ
functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨   2
functional_1/flatten/Const╨
functional_1/flatten/ReshapeReshape/functional_1/average_pooling2d/AvgPool:output:0#functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:         ╨2
functional_1/flatten/Reshape▄
/functional_1/dense_1_ee_1/MatMul/ReadVariableOpReadVariableOp8functional_1_dense_1_ee_1_matmul_readvariableop_resource*
_output_shapes
:	╨
*
dtype021
/functional_1/dense_1_ee_1/MatMul/ReadVariableOpр
 functional_1/dense_1_ee_1/MatMulMatMul%functional_1/flatten/Reshape:output:07functional_1/dense_1_ee_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2"
 functional_1/dense_1_ee_1/MatMul┌
0functional_1/dense_1_ee_1/BiasAdd/ReadVariableOpReadVariableOp9functional_1_dense_1_ee_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype022
0functional_1/dense_1_ee_1/BiasAdd/ReadVariableOpщ
!functional_1/dense_1_ee_1/BiasAddBiasAdd*functional_1/dense_1_ee_1/MatMul:product:08functional_1/dense_1_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2#
!functional_1/dense_1_ee_1/BiasAddп
!functional_1/dense_1_ee_1/SoftmaxSoftmax*functional_1/dense_1_ee_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2#
!functional_1/dense_1_ee_1/Softmax█
/functional_1/dense_2_ee_1/MatMul/ReadVariableOpReadVariableOp8functional_1_dense_2_ee_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype021
/functional_1/dense_2_ee_1/MatMul/ReadVariableOpц
 functional_1/dense_2_ee_1/MatMulMatMul+functional_1/dense_1_ee_1/Softmax:softmax:07functional_1/dense_2_ee_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 functional_1/dense_2_ee_1/MatMul┌
0functional_1/dense_2_ee_1/BiasAdd/ReadVariableOpReadVariableOp9functional_1_dense_2_ee_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0functional_1/dense_2_ee_1/BiasAdd/ReadVariableOpщ
!functional_1/dense_2_ee_1/BiasAddBiasAdd*functional_1/dense_2_ee_1/MatMul:product:08functional_1/dense_2_ee_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!functional_1/dense_2_ee_1/BiasAddп
!functional_1/dense_2_ee_1/SoftmaxSoftmax*functional_1/dense_2_ee_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2#
!functional_1/dense_2_ee_1/Softmax
IdentityIdentity+functional_1/dense_2_ee_1/Softmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	:::::::::::::::::::::::::::::::::::::Y U
0
_output_shapes
:         т	
!
_user_specified_name	input_1
╙
Й
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14773

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣:::::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
у
Щ
,__inference_functional_1_layer_call_fn_13594
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *<
_read_only_resource_inputs
	
 !"#$*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_135192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         т	
!
_user_specified_name	input_1
╚
a
B__inference_dropout_layer_call_and_return_conditional_losses_14551

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         ё2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         ё*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╟
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ё2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ё2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ё2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ё:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
┘
c
G__inference_activation_4_layer_call_and_return_conditional_losses_15175

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╣2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
┘
c
G__inference_activation_3_layer_call_and_return_conditional_losses_15006

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╣2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
┤
п
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_13251

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
:::O K
'
_output_shapes
:         

 
_user_specified_nameinputs
ъ
Ш
,__inference_functional_1_layer_call_fn_14370

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identityИвStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_137112
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
╞
л
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14586

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╙
ж
3__inference_batch_normalization_layer_call_fn_14681

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_127992
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╫
и
5__inference_batch_normalization_2_layer_call_fn_14988

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_130082
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╨
з
?__inference_pw_2_layer_call_and_return_conditional_losses_15033

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2	
BiasAdd╗
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╣:::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Д
н
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13008

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Л
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_12617

inputs
identity╢
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЗ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
∙
Ч
$__inference_bn_1_layer_call_fn_14516

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_120832
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ш
Й
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15080

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╖
п
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_15202

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╨
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╨:::P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
╒
ж
3__inference_batch_normalization_layer_call_fn_14694

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_128172
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╖
Ь
?__inference_bn_1_layer_call_and_return_conditional_losses_12083

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
э
Щ
,__inference_functional_1_layer_call_fn_13786
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identityИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_137112
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         т	
!
_user_specified_name	input_1
└
|
'__inference_dw_ee_1_layer_call_fn_12611

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dw_ee_1_layer_call_and_return_conditional_losses_126012
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╝
H
,__inference_activation_1_layer_call_fn_14704

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_128582
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
у
Б
,__inference_dense_2_ee_1_layer_call_fn_15231

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_132512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
┤
п
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_15222

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
:::O K
'
_output_shapes
:         

 
_user_specified_nameinputs
Щ
ж
3__inference_batch_normalization_layer_call_fn_14630

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_122412
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ч
ж
3__inference_batch_normalization_layer_call_fn_14617

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_122102
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╥
й
A__inference_conv_1_layer_call_and_return_conditional_losses_12643

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ё2	
BiasAdd╗
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         т	:::X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
╝
H
,__inference_activation_4_layer_call_fn_15180

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_131852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Д
н
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14755

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
щ
`
B__inference_dropout_layer_call_and_return_conditional_losses_14556

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         ё2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ё2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         ё:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14819

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╝
H
,__inference_activation_3_layer_call_fn_15011

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_130672
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╜
Р
#__inference_signature_wrapper_13883
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_120212
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:         т	
!
_user_specified_name	input_1
║
y
$__inference_dw_1_layer_call_fn_12148

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dw_1_layer_call_and_return_conditional_losses_121382
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
йШ
є
!__inference__traced_restore_15513
file_prefix"
assignvariableop_conv_1_kernel"
assignvariableop_1_conv_1_bias!
assignvariableop_2_bn_1_gamma 
assignvariableop_3_bn_1_beta'
#assignvariableop_4_bn_1_moving_mean+
'assignvariableop_5_bn_1_moving_variance,
(assignvariableop_6_dw_1_depthwise_kernel 
assignvariableop_7_dw_1_bias0
,assignvariableop_8_batch_normalization_gamma/
+assignvariableop_9_batch_normalization_beta7
3assignvariableop_10_batch_normalization_moving_mean;
7assignvariableop_11_batch_normalization_moving_variance#
assignvariableop_12_pw_1_kernel!
assignvariableop_13_pw_1_bias3
/assignvariableop_14_batch_normalization_1_gamma2
.assignvariableop_15_batch_normalization_1_beta9
5assignvariableop_16_batch_normalization_1_moving_mean=
9assignvariableop_17_batch_normalization_1_moving_variance-
)assignvariableop_18_dw_2_depthwise_kernel!
assignvariableop_19_dw_2_bias3
/assignvariableop_20_batch_normalization_2_gamma2
.assignvariableop_21_batch_normalization_2_beta9
5assignvariableop_22_batch_normalization_2_moving_mean=
9assignvariableop_23_batch_normalization_2_moving_variance#
assignvariableop_24_pw_2_kernel!
assignvariableop_25_pw_2_bias3
/assignvariableop_26_batch_normalization_3_gamma2
.assignvariableop_27_batch_normalization_3_beta9
5assignvariableop_28_batch_normalization_3_moving_mean=
9assignvariableop_29_batch_normalization_3_moving_variance0
,assignvariableop_30_dw_ee_1_depthwise_kernel$
 assignvariableop_31_dw_ee_1_bias+
'assignvariableop_32_dense_1_ee_1_kernel)
%assignvariableop_33_dense_1_ee_1_bias+
'assignvariableop_34_dense_2_ee_1_kernel)
%assignvariableop_35_dense_2_ee_1_bias
identity_37ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╥
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*▐
value╘B╤%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╪
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesч
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*к
_output_shapesЧ
Ф:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1г
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2в
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3б
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4и
AssignVariableOp_4AssignVariableOp#assignvariableop_4_bn_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5м
AssignVariableOp_5AssignVariableOp'assignvariableop_5_bn_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6н
AssignVariableOp_6AssignVariableOp(assignvariableop_6_dw_1_depthwise_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7б
AssignVariableOp_7AssignVariableOpassignvariableop_7_dw_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▒
AssignVariableOp_8AssignVariableOp,assignvariableop_8_batch_normalization_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9░
AssignVariableOp_9AssignVariableOp+assignvariableop_9_batch_normalization_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╗
AssignVariableOp_10AssignVariableOp3assignvariableop_10_batch_normalization_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┐
AssignVariableOp_11AssignVariableOp7assignvariableop_11_batch_normalization_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12з
AssignVariableOp_12AssignVariableOpassignvariableop_12_pw_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13е
AssignVariableOp_13AssignVariableOpassignvariableop_13_pw_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╖
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╢
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╜
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17┴
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18▒
AssignVariableOp_18AssignVariableOp)assignvariableop_18_dw_2_depthwise_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19е
AssignVariableOp_19AssignVariableOpassignvariableop_19_dw_2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╖
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_2_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╢
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_2_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╜
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_2_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23┴
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_2_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24з
AssignVariableOp_24AssignVariableOpassignvariableop_24_pw_2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25е
AssignVariableOp_25AssignVariableOpassignvariableop_25_pw_2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╖
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_3_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╢
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_3_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╜
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_3_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29┴
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_3_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30┤
AssignVariableOp_30AssignVariableOp,assignvariableop_30_dw_ee_1_depthwise_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31и
AssignVariableOp_31AssignVariableOp assignvariableop_31_dw_ee_1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32п
AssignVariableOp_32AssignVariableOp'assignvariableop_32_dense_1_ee_1_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33н
AssignVariableOp_33AssignVariableOp%assignvariableop_33_dense_1_ee_1_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34п
AssignVariableOp_34AssignVariableOp'assignvariableop_34_dense_2_ee_1_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35н
AssignVariableOp_35AssignVariableOp%assignvariableop_35_dense_2_ee_1_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_359
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЎ
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_36щ
Identity_37IdentityIdentity_36:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_37"#
identity_37Identity_37:output:0*з
_input_shapesХ
Т: ::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╖
Ь
?__inference_bn_1_layer_call_and_return_conditional_losses_14485

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
║
^
B__inference_flatten_layer_call_and_return_conditional_losses_13205

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ╨2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ╨2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
р
Ш
,__inference_functional_1_layer_call_fn_14293

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *<
_read_only_resource_inputs
	
 !"#$*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_135192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┴
_input_shapesп
м:         т	::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         т	
 
_user_specified_nameinputs
┘
и
5__inference_batch_normalization_3_layer_call_fn_15170

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_131442
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╝	
e
__inference_loss_fn_1_152535
1kernel_regularizer_square_readvariableop_resource
identityИ╬
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
╝	
e
__inference_loss_fn_2_152645
1kernel_regularizer_square_readvariableop_resource
identityИ╬
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
║
^
B__inference_flatten_layer_call_and_return_conditional_losses_15186

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ╨   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ╨2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ╨2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╙
Й
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_13026

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣:::::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╫
a
E__inference_activation_layer_call_and_return_conditional_losses_12737

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ё2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ё:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
╞
л
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12210

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╫
и
5__inference_batch_normalization_3_layer_call_fn_15157

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ╣*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_131262
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
Д
н
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15126

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
В
л
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12799

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╤
З
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12817

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣:::::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╨
з
?__inference_pw_2_layer_call_and_return_conditional_losses_13091

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ╣2	
BiasAdd╗
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mulm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ╣:::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12441

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╙
Й
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14975

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣:::::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15062

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╖
Ч
$__inference_bn_1_layer_call_fn_14465

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_126962
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ё::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
┘
c
G__inference_activation_2_layer_call_and_return_conditional_losses_12976

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ╣2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╣:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
З
°
?__inference_bn_1_layer_call_and_return_conditional_losses_14503

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З
°
?__inference_bn_1_layer_call_and_return_conditional_losses_12114

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           :::::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
є
Ь
?__inference_bn_1_layer_call_and_return_conditional_losses_12678

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ё:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ё::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12314

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╤
З
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14668

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╦
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣:::::X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╚
н
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14893

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ж
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Д
н
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14957

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┘
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:         ╣:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3 
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueН
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Х
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:         ╣2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ╣::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:         ╣
 
_user_specified_nameinputs
╝	
e
__inference_loss_fn_0_152425
1kernel_regularizer_square_readvariableop_resource
identityИ╬
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:
*
dtype02*
(kernel/Regularizer/Square/ReadVariableOpг
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
kernel/Regularizer/SquareН
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
kernel/Regularizer/ConstЪ
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/Sumy
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82
kernel/Regularizer/mul/xЬ
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
kernel/Regularizer/mul]
IdentityIdentitykernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
╡
Ч
$__inference_bn_1_layer_call_fn_14452

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ё*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_126782
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ё2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ё::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ё
 
_user_specified_nameinputs
х
Б
,__inference_dense_1_ee_1_layer_call_fn_15211

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_132242
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╨::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
а
C
'__inference_flatten_layer_call_fn_15191

inputs
identity┴
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_132052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╨2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╕
serving_defaultд
D
input_19
serving_default_input_1:0         т	@
dense_2_ee_10
StatefulPartitionedCall:0         tensorflow/serving/predict:х╚
╙┤
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
layer-19
layer_with_weights-11
layer-20
layer_with_weights-12
layer-21
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Э__call__
+Ю&call_and_return_all_conditional_losses
Я_default_save_signature"Фо
_tf_keras_networkўн{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [10, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "dw_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "dw_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["dw_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "pw_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pw_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["pw_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "dw_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "dw_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["dw_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "pw_2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pw_2", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["pw_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "dw_ee_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "dw_ee_1", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [6, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [6, 1]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["dw_ee_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1_ee_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1_ee_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2_ee_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2_ee_1", "inbound_nodes": [[["dense_1_ee_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2_ee_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1250, 1, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [10, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "dw_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "dw_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["dw_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "pw_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pw_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["pw_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "dw_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "dw_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["dw_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "pw_2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pw_2", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["pw_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "dw_ee_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "dw_ee_1", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [6, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [6, 1]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["dw_ee_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1_ee_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1_ee_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2_ee_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2_ee_1", "inbound_nodes": [[["dense_1_ee_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2_ee_1", 0, 0]]}}}
а
#_self_saveable_object_factories"°
_tf_keras_input_layer╪{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1250, 1, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
╧


kernel
bias
#_self_saveable_object_factories
 regularization_losses
!trainable_variables
"	variables
#	keras_api
а__call__
+б&call_and_return_all_conditional_losses"Г	
_tf_keras_layerщ{"class_name": "Conv2D", "name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [10, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1250, 1, 1]}}
╜	
$axis
	%gamma
&beta
'moving_mean
(moving_variance
#)_self_saveable_object_factories
*regularization_losses
+trainable_variables
,	variables
-	keras_api
в__call__
+г&call_and_return_all_conditional_losses"┬
_tf_keras_layerи{"class_name": "BatchNormalization", "name": "bn_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 625, 1, 4]}}
°
#._self_saveable_object_factories
/regularization_losses
0trainable_variables
1	variables
2	keras_api
д__call__
+е&call_and_return_all_conditional_losses"┬
_tf_keras_layerи{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
И
#3_self_saveable_object_factories
4regularization_losses
5trainable_variables
6	variables
7	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"╥
_tf_keras_layer╕{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
╡

8depthwise_kernel
9bias
#:_self_saveable_object_factories
;regularization_losses
<trainable_variables
=	variables
>	keras_api
и__call__
+й&call_and_return_all_conditional_losses"▀
_tf_keras_layer┼{"class_name": "DepthwiseConv2D", "name": "dw_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dw_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 625, 1, 4]}}
█	
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
к__call__
+л&call_and_return_all_conditional_losses"р
_tf_keras_layer╞{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 313, 1, 4]}}
№
#I_self_saveable_object_factories
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
м__call__
+н&call_and_return_all_conditional_losses"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
╔


Nkernel
Obias
#P_self_saveable_object_factories
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
о__call__
+п&call_and_return_all_conditional_losses"¤
_tf_keras_layerу{"class_name": "Conv2D", "name": "pw_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pw_1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 313, 1, 4]}}
▀	
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
#Z_self_saveable_object_factories
[regularization_losses
\trainable_variables
]	variables
^	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 313, 1, 8]}}
№
#__self_saveable_object_factories
`regularization_losses
atrainable_variables
b	variables
c	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
╡

ddepthwise_kernel
ebias
#f_self_saveable_object_factories
gregularization_losses
htrainable_variables
i	variables
j	keras_api
┤__call__
+╡&call_and_return_all_conditional_losses"▀
_tf_keras_layer┼{"class_name": "DepthwiseConv2D", "name": "dw_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dw_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 313, 1, 8]}}
▀	
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
#p_self_saveable_object_factories
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
╢__call__
+╖&call_and_return_all_conditional_losses"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 313, 1, 8]}}
№
#u_self_saveable_object_factories
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
╕__call__
+╣&call_and_return_all_conditional_losses"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
╩


zkernel
{bias
#|_self_saveable_object_factories
}regularization_losses
~trainable_variables
	variables
А	keras_api
║__call__
+╗&call_and_return_all_conditional_losses"¤
_tf_keras_layerу{"class_name": "Conv2D", "name": "pw_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pw_2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 313, 1, 8]}}
щ	
	Бaxis

Вgamma
	Гbeta
Дmoving_mean
Еmoving_variance
$Ж_self_saveable_object_factories
Зregularization_losses
Иtrainable_variables
Й	variables
К	keras_api
╝__call__
+╜&call_and_return_all_conditional_losses"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 313, 1, 8]}}
Б
$Л_self_saveable_object_factories
Мregularization_losses
Нtrainable_variables
О	variables
П	keras_api
╛__call__
+┐&call_and_return_all_conditional_losses"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
└

Рdepthwise_kernel
	Сbias
$Т_self_saveable_object_factories
Уregularization_losses
Фtrainable_variables
Х	variables
Ц	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"у
_tf_keras_layer╔{"class_name": "DepthwiseConv2D", "name": "dw_ee_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dw_ee_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 313, 1, 8]}}
│
$Ч_self_saveable_object_factories
Шregularization_losses
Щtrainable_variables
Ъ	variables
Ы	keras_api
┬__call__
+├&call_and_return_all_conditional_losses"°
_tf_keras_layer▐{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [6, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [6, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
О
$Ь_self_saveable_object_factories
Эregularization_losses
Юtrainable_variables
Я	variables
а	keras_api
─__call__
+┼&call_and_return_all_conditional_losses"╙
_tf_keras_layer╣{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
и
бkernel
	вbias
$г_self_saveable_object_factories
дregularization_losses
еtrainable_variables
ж	variables
з	keras_api
╞__call__
+╟&call_and_return_all_conditional_losses"╒
_tf_keras_layer╗{"class_name": "Dense", "name": "dense_1_ee_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1_ee_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 208}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 208]}}
е
иkernel
	йbias
$к_self_saveable_object_factories
лregularization_losses
мtrainable_variables
н	variables
о	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses"╥
_tf_keras_layer╕{"class_name": "Dense", "name": "dense_2_ee_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2_ee_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
8
╩0
╦1
╠2"
trackable_list_wrapper
ю
0
1
%2
&3
84
95
@6
A7
N8
O9
V10
W11
d12
e13
l14
m15
z16
{17
В18
Г19
Р20
С21
б22
в23
и24
й25"
trackable_list_wrapper
└
0
1
%2
&3
'4
(5
86
97
@8
A9
B10
C11
N12
O13
V14
W15
X16
Y17
d18
e19
l20
m21
n22
o23
z24
{25
В26
Г27
Д28
Е29
Р30
С31
б32
в33
и34
й35"
trackable_list_wrapper
╙
пmetrics
regularization_losses
 ░layer_regularization_losses
▒non_trainable_variables
trainable_variables
▓layer_metrics
│layers
	variables
Э__call__
Я_default_save_signature
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
-
═serving_default"
signature_map
 "
trackable_dict_wrapper
':%
2conv_1/kernel
:2conv_1/bias
 "
trackable_dict_wrapper
(
╩0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╡
┤metrics
 regularization_losses
 ╡layer_regularization_losses
╢non_trainable_variables
!trainable_variables
╖layer_metrics
╕layers
"	variables
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2
bn_1/gamma
:2	bn_1/beta
 : (2bn_1/moving_mean
$:" (2bn_1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
╡
╣metrics
*regularization_losses
 ║layer_regularization_losses
╗non_trainable_variables
+trainable_variables
╝layer_metrics
╜layers
,	variables
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╛metrics
/regularization_losses
 ┐layer_regularization_losses
└non_trainable_variables
0trainable_variables
┴layer_metrics
┬layers
1	variables
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
├metrics
4regularization_losses
 ─layer_regularization_losses
┼non_trainable_variables
5trainable_variables
╞layer_metrics
╟layers
6	variables
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
/:-2dw_1/depthwise_kernel
:2	dw_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
╡
╚metrics
;regularization_losses
 ╔layer_regularization_losses
╩non_trainable_variables
<trainable_variables
╦layer_metrics
╠layers
=	variables
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
<
@0
A1
B2
C3"
trackable_list_wrapper
╡
═metrics
Eregularization_losses
 ╬layer_regularization_losses
╧non_trainable_variables
Ftrainable_variables
╨layer_metrics
╤layers
G	variables
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╥metrics
Jregularization_losses
 ╙layer_regularization_losses
╘non_trainable_variables
Ktrainable_variables
╒layer_metrics
╓layers
L	variables
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
%:#2pw_1/kernel
:2	pw_1/bias
 "
trackable_dict_wrapper
(
╦0"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
╡
╫metrics
Qregularization_losses
 ╪layer_regularization_losses
┘non_trainable_variables
Rtrainable_variables
┌layer_metrics
█layers
S	variables
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
<
V0
W1
X2
Y3"
trackable_list_wrapper
╡
▄metrics
[regularization_losses
 ▌layer_regularization_losses
▐non_trainable_variables
\trainable_variables
▀layer_metrics
рlayers
]	variables
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
сmetrics
`regularization_losses
 тlayer_regularization_losses
уnon_trainable_variables
atrainable_variables
фlayer_metrics
хlayers
b	variables
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
/:-2dw_2/depthwise_kernel
:2	dw_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
╡
цmetrics
gregularization_losses
 чlayer_regularization_losses
шnon_trainable_variables
htrainable_variables
щlayer_metrics
ъlayers
i	variables
┤__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_2/gamma
(:&2batch_normalization_2/beta
1:/ (2!batch_normalization_2/moving_mean
5:3 (2%batch_normalization_2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
<
l0
m1
n2
o3"
trackable_list_wrapper
╡
ыmetrics
qregularization_losses
 ьlayer_regularization_losses
эnon_trainable_variables
rtrainable_variables
юlayer_metrics
яlayers
s	variables
╢__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Ёmetrics
vregularization_losses
 ёlayer_regularization_losses
Єnon_trainable_variables
wtrainable_variables
єlayer_metrics
Їlayers
x	variables
╕__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
%:#2pw_2/kernel
:2	pw_2/bias
 "
trackable_dict_wrapper
(
╠0"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
╡
їmetrics
}regularization_losses
 Ўlayer_regularization_losses
ўnon_trainable_variables
~trainable_variables
°layer_metrics
∙layers
	variables
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_3/gamma
(:&2batch_normalization_3/beta
1:/ (2!batch_normalization_3/moving_mean
5:3 (2%batch_normalization_3/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
@
В0
Г1
Д2
Е3"
trackable_list_wrapper
╕
·metrics
Зregularization_losses
 √layer_regularization_losses
№non_trainable_variables
Иtrainable_variables
¤layer_metrics
■layers
Й	variables
╝__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
 metrics
Мregularization_losses
 Аlayer_regularization_losses
Бnon_trainable_variables
Нtrainable_variables
Вlayer_metrics
Гlayers
О	variables
╛__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
2:02dw_ee_1/depthwise_kernel
:2dw_ee_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
╕
Дmetrics
Уregularization_losses
 Еlayer_regularization_losses
Жnon_trainable_variables
Фtrainable_variables
Зlayer_metrics
Иlayers
Х	variables
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Йmetrics
Шregularization_losses
 Кlayer_regularization_losses
Лnon_trainable_variables
Щtrainable_variables
Мlayer_metrics
Нlayers
Ъ	variables
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Оmetrics
Эregularization_losses
 Пlayer_regularization_losses
Рnon_trainable_variables
Юtrainable_variables
Сlayer_metrics
Тlayers
Я	variables
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
&:$	╨
2dense_1_ee_1/kernel
:
2dense_1_ee_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
╕
Уmetrics
дregularization_losses
 Фlayer_regularization_losses
Хnon_trainable_variables
еtrainable_variables
Цlayer_metrics
Чlayers
ж	variables
╞__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses"
_generic_user_object
%:#
2dense_2_ee_1/kernel
:2dense_2_ee_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
и0
й1"
trackable_list_wrapper
0
и0
й1"
trackable_list_wrapper
╕
Шmetrics
лregularization_losses
 Щlayer_regularization_losses
Ъnon_trainable_variables
мtrainable_variables
Ыlayer_metrics
Ьlayers
н	variables
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
h
'0
(1
B2
C3
X4
Y5
n6
o7
Д8
Е9"
trackable_list_wrapper
 "
trackable_dict_wrapper
╞
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
(
╩0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
╦0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
╠0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
■2√
,__inference_functional_1_layer_call_fn_13786
,__inference_functional_1_layer_call_fn_14370
,__inference_functional_1_layer_call_fn_13594
,__inference_functional_1_layer_call_fn_14293└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
G__inference_functional_1_layer_call_and_return_conditional_losses_14216
G__inference_functional_1_layer_call_and_return_conditional_losses_14058
G__inference_functional_1_layer_call_and_return_conditional_losses_13401
G__inference_functional_1_layer_call_and_return_conditional_losses_13286└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ч2ф
 __inference__wrapped_model_12021┐
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк */в,
*К'
input_1         т	
╨2═
&__inference_conv_1_layer_call_fn_14401в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_conv_1_layer_call_and_return_conditional_losses_14392в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
$__inference_bn_1_layer_call_fn_14516
$__inference_bn_1_layer_call_fn_14529
$__inference_bn_1_layer_call_fn_14465
$__inference_bn_1_layer_call_fn_14452┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╛2╗
?__inference_bn_1_layer_call_and_return_conditional_losses_14421
?__inference_bn_1_layer_call_and_return_conditional_losses_14485
?__inference_bn_1_layer_call_and_return_conditional_losses_14503
?__inference_bn_1_layer_call_and_return_conditional_losses_14439┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
*__inference_activation_layer_call_fn_14539в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_activation_layer_call_and_return_conditional_losses_14534в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
М2Й
'__inference_dropout_layer_call_fn_14566
'__inference_dropout_layer_call_fn_14561┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┬2┐
B__inference_dropout_layer_call_and_return_conditional_losses_14551
B__inference_dropout_layer_call_and_return_conditional_losses_14556┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Г2А
$__inference_dw_1_layer_call_fn_12148╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Ю2Ы
?__inference_dw_1_layer_call_and_return_conditional_losses_12138╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
О2Л
3__inference_batch_normalization_layer_call_fn_14630
3__inference_batch_normalization_layer_call_fn_14617
3__inference_batch_normalization_layer_call_fn_14681
3__inference_batch_normalization_layer_call_fn_14694┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
·2ў
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14586
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14650
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14604
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14668┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
,__inference_activation_1_layer_call_fn_14704в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_activation_1_layer_call_and_return_conditional_losses_14699в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_pw_1_layer_call_fn_14735в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_pw_1_layer_call_and_return_conditional_losses_14726в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ц2У
5__inference_batch_normalization_1_layer_call_fn_14786
5__inference_batch_normalization_1_layer_call_fn_14799
5__inference_batch_normalization_1_layer_call_fn_14850
5__inference_batch_normalization_1_layer_call_fn_14863┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14773
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14755
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14819
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14837┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
,__inference_activation_2_layer_call_fn_14873в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_activation_2_layer_call_and_return_conditional_losses_14868в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Г2А
$__inference_dw_2_layer_call_fn_12379╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Ю2Ы
?__inference_dw_2_layer_call_and_return_conditional_losses_12369╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Ц2У
5__inference_batch_normalization_2_layer_call_fn_14924
5__inference_batch_normalization_2_layer_call_fn_14937
5__inference_batch_normalization_2_layer_call_fn_14988
5__inference_batch_normalization_2_layer_call_fn_15001┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14893
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14911
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14957
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14975┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
,__inference_activation_3_layer_call_fn_15011в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_activation_3_layer_call_and_return_conditional_losses_15006в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_pw_2_layer_call_fn_15042в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_pw_2_layer_call_and_return_conditional_losses_15033в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ц2У
5__inference_batch_normalization_3_layer_call_fn_15157
5__inference_batch_normalization_3_layer_call_fn_15093
5__inference_batch_normalization_3_layer_call_fn_15106
5__inference_batch_normalization_3_layer_call_fn_15170┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15080
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15062
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15126
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15144┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
,__inference_activation_4_layer_call_fn_15180в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_activation_4_layer_call_and_return_conditional_losses_15175в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ж2Г
'__inference_dw_ee_1_layer_call_fn_12611╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
б2Ю
B__inference_dw_ee_1_layer_call_and_return_conditional_losses_12601╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Щ2Ц
1__inference_average_pooling2d_layer_call_fn_12623р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
┤2▒
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_12617р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╤2╬
'__inference_flatten_layer_call_fn_15191в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_15186в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_dense_1_ee_1_layer_call_fn_15211в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_15202в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_dense_2_ee_1_layer_call_fn_15231в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_15222в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▓2п
__inference_loss_fn_0_15242П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_1_15253П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_2_15264П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
2B0
#__inference_signature_wrapper_13883input_1═
 __inference__wrapped_model_12021и.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвий9в6
/в,
*К'
input_1         т	
к ";к8
6
dense_2_ee_1&К#
dense_2_ee_1         ╡
G__inference_activation_1_layer_call_and_return_conditional_losses_14699j8в5
.в+
)К&
inputs         ╣
к ".в+
$К!
0         ╣
Ъ Н
,__inference_activation_1_layer_call_fn_14704]8в5
.в+
)К&
inputs         ╣
к "!К         ╣╡
G__inference_activation_2_layer_call_and_return_conditional_losses_14868j8в5
.в+
)К&
inputs         ╣
к ".в+
$К!
0         ╣
Ъ Н
,__inference_activation_2_layer_call_fn_14873]8в5
.в+
)К&
inputs         ╣
к "!К         ╣╡
G__inference_activation_3_layer_call_and_return_conditional_losses_15006j8в5
.в+
)К&
inputs         ╣
к ".в+
$К!
0         ╣
Ъ Н
,__inference_activation_3_layer_call_fn_15011]8в5
.в+
)К&
inputs         ╣
к "!К         ╣╡
G__inference_activation_4_layer_call_and_return_conditional_losses_15175j8в5
.в+
)К&
inputs         ╣
к ".в+
$К!
0         ╣
Ъ Н
,__inference_activation_4_layer_call_fn_15180]8в5
.в+
)К&
inputs         ╣
к "!К         ╣│
E__inference_activation_layer_call_and_return_conditional_losses_14534j8в5
.в+
)К&
inputs         ё
к ".в+
$К!
0         ё
Ъ Л
*__inference_activation_layer_call_fn_14539]8в5
.в+
)К&
inputs         ё
к "!К         ёя
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_12617ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_average_pooling2d_layer_call_fn_12623СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╚
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14755tVWXY<в9
2в/
)К&
inputs         ╣
p
к ".в+
$К!
0         ╣
Ъ ╚
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14773tVWXY<в9
2в/
)К&
inputs         ╣
p 
к ".в+
$К!
0         ╣
Ъ ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14819ЦVWXYMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ы
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_14837ЦVWXYMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ а
5__inference_batch_normalization_1_layer_call_fn_14786gVWXY<в9
2в/
)К&
inputs         ╣
p
к "!К         ╣а
5__inference_batch_normalization_1_layer_call_fn_14799gVWXY<в9
2в/
)К&
inputs         ╣
p 
к "!К         ╣├
5__inference_batch_normalization_1_layer_call_fn_14850ЙVWXYMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ├
5__inference_batch_normalization_1_layer_call_fn_14863ЙVWXYMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14893ЦlmnoMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ы
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14911ЦlmnoMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╚
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14957tlmno<в9
2в/
)К&
inputs         ╣
p
к ".в+
$К!
0         ╣
Ъ ╚
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_14975tlmno<в9
2в/
)К&
inputs         ╣
p 
к ".в+
$К!
0         ╣
Ъ ├
5__inference_batch_normalization_2_layer_call_fn_14924ЙlmnoMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ├
5__inference_batch_normalization_2_layer_call_fn_14937ЙlmnoMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           а
5__inference_batch_normalization_2_layer_call_fn_14988glmno<в9
2в/
)К&
inputs         ╣
p
к "!К         ╣а
5__inference_batch_normalization_2_layer_call_fn_15001glmno<в9
2в/
)К&
inputs         ╣
p 
к "!К         ╣я
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15062ЪВГДЕMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ я
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15080ЪВГДЕMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╠
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15126xВГДЕ<в9
2в/
)К&
inputs         ╣
p
к ".в+
$К!
0         ╣
Ъ ╠
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_15144xВГДЕ<в9
2в/
)К&
inputs         ╣
p 
к ".в+
$К!
0         ╣
Ъ ╟
5__inference_batch_normalization_3_layer_call_fn_15093НВГДЕMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ╟
5__inference_batch_normalization_3_layer_call_fn_15106НВГДЕMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           д
5__inference_batch_normalization_3_layer_call_fn_15157kВГДЕ<в9
2в/
)К&
inputs         ╣
p
к "!К         ╣д
5__inference_batch_normalization_3_layer_call_fn_15170kВГДЕ<в9
2в/
)К&
inputs         ╣
p 
к "!К         ╣щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14586Ц@ABCMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ щ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14604Ц@ABCMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╞
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14650t@ABC<в9
2в/
)К&
inputs         ╣
p
к ".в+
$К!
0         ╣
Ъ ╞
N__inference_batch_normalization_layer_call_and_return_conditional_losses_14668t@ABC<в9
2в/
)К&
inputs         ╣
p 
к ".в+
$К!
0         ╣
Ъ ┴
3__inference_batch_normalization_layer_call_fn_14617Й@ABCMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ┴
3__inference_batch_normalization_layer_call_fn_14630Й@ABCMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           Ю
3__inference_batch_normalization_layer_call_fn_14681g@ABC<в9
2в/
)К&
inputs         ╣
p
к "!К         ╣Ю
3__inference_batch_normalization_layer_call_fn_14694g@ABC<в9
2в/
)К&
inputs         ╣
p 
к "!К         ╣╖
?__inference_bn_1_layer_call_and_return_conditional_losses_14421t%&'(<в9
2в/
)К&
inputs         ё
p
к ".в+
$К!
0         ё
Ъ ╖
?__inference_bn_1_layer_call_and_return_conditional_losses_14439t%&'(<в9
2в/
)К&
inputs         ё
p 
к ".в+
$К!
0         ё
Ъ ┌
?__inference_bn_1_layer_call_and_return_conditional_losses_14485Ц%&'(MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ┌
?__inference_bn_1_layer_call_and_return_conditional_losses_14503Ц%&'(MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ П
$__inference_bn_1_layer_call_fn_14452g%&'(<в9
2в/
)К&
inputs         ё
p
к "!К         ёП
$__inference_bn_1_layer_call_fn_14465g%&'(<в9
2в/
)К&
inputs         ё
p 
к "!К         ё▓
$__inference_bn_1_layer_call_fn_14516Й%&'(MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ▓
$__inference_bn_1_layer_call_fn_14529Й%&'(MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           │
A__inference_conv_1_layer_call_and_return_conditional_losses_14392n8в5
.в+
)К&
inputs         т	
к ".в+
$К!
0         ё
Ъ Л
&__inference_conv_1_layer_call_fn_14401a8в5
.в+
)К&
inputs         т	
к "!К         ёк
G__inference_dense_1_ee_1_layer_call_and_return_conditional_losses_15202_бв0в-
&в#
!К
inputs         ╨
к "%в"
К
0         

Ъ В
,__inference_dense_1_ee_1_layer_call_fn_15211Rбв0в-
&в#
!К
inputs         ╨
к "К         
й
G__inference_dense_2_ee_1_layer_call_and_return_conditional_losses_15222^ий/в,
%в"
 К
inputs         

к "%в"
К
0         
Ъ Б
,__inference_dense_2_ee_1_layer_call_fn_15231Qий/в,
%в"
 К
inputs         

к "К         ┤
B__inference_dropout_layer_call_and_return_conditional_losses_14551n<в9
2в/
)К&
inputs         ё
p
к ".в+
$К!
0         ё
Ъ ┤
B__inference_dropout_layer_call_and_return_conditional_losses_14556n<в9
2в/
)К&
inputs         ё
p 
к ".в+
$К!
0         ё
Ъ М
'__inference_dropout_layer_call_fn_14561a<в9
2в/
)К&
inputs         ё
p
к "!К         ёМ
'__inference_dropout_layer_call_fn_14566a<в9
2в/
)К&
inputs         ё
p 
к "!К         ё╘
?__inference_dw_1_layer_call_and_return_conditional_losses_12138Р89IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ м
$__inference_dw_1_layer_call_fn_12148Г89IвF
?в<
:К7
inputs+                           
к "2К/+                           ╘
?__inference_dw_2_layer_call_and_return_conditional_losses_12369РdeIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ м
$__inference_dw_2_layer_call_fn_12379ГdeIвF
?в<
:К7
inputs+                           
к "2К/+                           ┘
B__inference_dw_ee_1_layer_call_and_return_conditional_losses_12601ТРСIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▒
'__inference_dw_ee_1_layer_call_fn_12611ЕРСIвF
?в<
:К7
inputs+                           
к "2К/+                           з
B__inference_flatten_layer_call_and_return_conditional_losses_15186a7в4
-в*
(К%
inputs         
к "&в#
К
0         ╨
Ъ 
'__inference_flatten_layer_call_fn_15191T7в4
-в*
(К%
inputs         
к "К         ╨ц
G__inference_functional_1_layer_call_and_return_conditional_losses_13286Ъ.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвийAв>
7в4
*К'
input_1         т	
p

 
к "%в"
К
0         
Ъ ц
G__inference_functional_1_layer_call_and_return_conditional_losses_13401Ъ.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвийAв>
7в4
*К'
input_1         т	
p 

 
к "%в"
К
0         
Ъ х
G__inference_functional_1_layer_call_and_return_conditional_losses_14058Щ.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвий@в=
6в3
)К&
inputs         т	
p

 
к "%в"
К
0         
Ъ х
G__inference_functional_1_layer_call_and_return_conditional_losses_14216Щ.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвий@в=
6в3
)К&
inputs         т	
p 

 
к "%в"
К
0         
Ъ ╛
,__inference_functional_1_layer_call_fn_13594Н.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвийAв>
7в4
*К'
input_1         т	
p

 
к "К         ╛
,__inference_functional_1_layer_call_fn_13786Н.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвийAв>
7в4
*К'
input_1         т	
p 

 
к "К         ╜
,__inference_functional_1_layer_call_fn_14293М.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвий@в=
6в3
)К&
inputs         т	
p

 
к "К         ╜
,__inference_functional_1_layer_call_fn_14370М.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвий@в=
6в3
)К&
inputs         т	
p 

 
к "К         :
__inference_loss_fn_0_15242в

в 
к "К :
__inference_loss_fn_1_15253Nв

в 
к "К :
__inference_loss_fn_2_15264zв

в 
к "К ▒
?__inference_pw_1_layer_call_and_return_conditional_losses_14726nNO8в5
.в+
)К&
inputs         ╣
к ".в+
$К!
0         ╣
Ъ Й
$__inference_pw_1_layer_call_fn_14735aNO8в5
.в+
)К&
inputs         ╣
к "!К         ╣▒
?__inference_pw_2_layer_call_and_return_conditional_losses_15033nz{8в5
.в+
)К&
inputs         ╣
к ".в+
$К!
0         ╣
Ъ Й
$__inference_pw_2_layer_call_fn_15042az{8в5
.в+
)К&
inputs         ╣
к "!К         ╣█
#__inference_signature_wrapper_13883│.%&'(89@ABCNOVWXYdelmnoz{ВГДЕРСбвийDвA
в 
:к7
5
input_1*К'
input_1         т	";к8
6
dense_2_ee_1&К#
dense_2_ee_1         