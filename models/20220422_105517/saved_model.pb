??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??

~
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer1/kernel
w
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*&
_output_shapes
:*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:*
dtype0
~
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer2/kernel
w
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*&
_output_shapes
:*
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:*
dtype0
~
layer5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer5/kernel
w
!layer5/kernel/Read/ReadVariableOpReadVariableOplayer5/kernel*&
_output_shapes
: *
dtype0
n
layer5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer5/bias
g
layer5/bias/Read/ReadVariableOpReadVariableOplayer5/bias*
_output_shapes
: *
dtype0
~
layer6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_namelayer6/kernel
w
!layer6/kernel/Read/ReadVariableOpReadVariableOplayer6/kernel*&
_output_shapes
:  *
dtype0
n
layer6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer6/bias
g
layer6/bias/Read/ReadVariableOpReadVariableOplayer6/bias*
_output_shapes
: *
dtype0
z
layer10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namelayer10/kernel
s
"layer10/kernel/Read/ReadVariableOpReadVariableOplayer10/kernel* 
_output_shapes
:
??*
dtype0
q
layer10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer10/bias
j
 layer10/bias/Read/ReadVariableOpReadVariableOplayer10/bias*
_output_shapes	
:?*
dtype0
z
layer11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namelayer11/kernel
s
"layer11/kernel/Read/ReadVariableOpReadVariableOplayer11/kernel* 
_output_shapes
:
??*
dtype0
q
layer11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer11/bias
j
 layer11/bias/Read/ReadVariableOpReadVariableOplayer11/bias*
_output_shapes	
:?*
dtype0
w
symbol/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namesymbol/kernel
p
!symbol/kernel/Read/ReadVariableOpReadVariableOpsymbol/kernel*
_output_shapes
:	?*
dtype0
n
symbol/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesymbol/bias
g
symbol/bias/Read/ReadVariableOpReadVariableOpsymbol/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/layer1/kernel/m
?
(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer1/bias/m
u
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/layer2/kernel/m
?
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/m
u
&Adam/layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/layer5/kernel/m
?
(Adam/layer5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer5/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/layer5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer5/bias/m
u
&Adam/layer5/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer5/bias/m*
_output_shapes
: *
dtype0
?
Adam/layer6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/layer6/kernel/m
?
(Adam/layer6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer6/kernel/m*&
_output_shapes
:  *
dtype0
|
Adam/layer6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer6/bias/m
u
&Adam/layer6/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer6/bias/m*
_output_shapes
: *
dtype0
?
Adam/layer10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/layer10/kernel/m
?
)Adam/layer10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer10/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/layer10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/layer10/bias/m
x
'Adam/layer10/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer10/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/layer11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/layer11/kernel/m
?
)Adam/layer11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer11/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/layer11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/layer11/bias/m
x
'Adam/layer11/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer11/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/symbol/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/symbol/kernel/m
~
(Adam/symbol/kernel/m/Read/ReadVariableOpReadVariableOpAdam/symbol/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/symbol/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/symbol/bias/m
u
&Adam/symbol/bias/m/Read/ReadVariableOpReadVariableOpAdam/symbol/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/layer1/kernel/v
?
(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer1/bias/v
u
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/layer2/kernel/v
?
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/v
u
&Adam/layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/layer5/kernel/v
?
(Adam/layer5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer5/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/layer5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer5/bias/v
u
&Adam/layer5/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer5/bias/v*
_output_shapes
: *
dtype0
?
Adam/layer6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameAdam/layer6/kernel/v
?
(Adam/layer6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer6/kernel/v*&
_output_shapes
:  *
dtype0
|
Adam/layer6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer6/bias/v
u
&Adam/layer6/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer6/bias/v*
_output_shapes
: *
dtype0
?
Adam/layer10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/layer10/kernel/v
?
)Adam/layer10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer10/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/layer10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/layer10/bias/v
x
'Adam/layer10/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer10/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/layer11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/layer11/kernel/v
?
)Adam/layer11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer11/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/layer11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/layer11/bias/v
x
'Adam/layer11/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer11/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/symbol/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdam/symbol/kernel/v
~
(Adam/symbol/kernel/v/Read/ReadVariableOpReadVariableOpAdam/symbol/kernel/v*
_output_shapes
:	?*
dtype0
|
Adam/symbol/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/symbol/bias/v
u
&Adam/symbol/bias/v/Read/ReadVariableOpReadVariableOpAdam/symbol/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?U
value?UB?U B?U
?
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
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
 regularization_losses
!trainable_variables
"	variables
#	keras_api
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
R
4regularization_losses
5trainable_variables
6	variables
7	keras_api
R
8regularization_losses
9trainable_variables
:	variables
;	keras_api
R
<regularization_losses
=trainable_variables
>	variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem?m?m?m?(m?)m?.m?/m?@m?Am?Fm?Gm?Lm?Mm?v?v?v?v?(v?)v?.v?/v?@v?Av?Fv?Gv?Lv?Mv?
 
f
0
1
2
3
(4
)5
.6
/7
@8
A9
F10
G11
L12
M13
f
0
1
2
3
(4
)5
.6
/7
@8
A9
F10
G11
L12
M13
?
Wlayer_regularization_losses
regularization_losses

Xlayers
trainable_variables
	variables
Ynon_trainable_variables
Zlayer_metrics
[metrics
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
\layer_regularization_losses
regularization_losses

]layers
trainable_variables
	variables
^non_trainable_variables
_layer_metrics
`metrics
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
alayer_regularization_losses
regularization_losses

blayers
trainable_variables
	variables
cnon_trainable_variables
dlayer_metrics
emetrics
 
 
 
?
flayer_regularization_losses
 regularization_losses

glayers
!trainable_variables
"	variables
hnon_trainable_variables
ilayer_metrics
jmetrics
 
 
 
?
klayer_regularization_losses
$regularization_losses

llayers
%trainable_variables
&	variables
mnon_trainable_variables
nlayer_metrics
ometrics
YW
VARIABLE_VALUElayer5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?
player_regularization_losses
*regularization_losses

qlayers
+trainable_variables
,	variables
rnon_trainable_variables
slayer_metrics
tmetrics
YW
VARIABLE_VALUElayer6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
ulayer_regularization_losses
0regularization_losses

vlayers
1trainable_variables
2	variables
wnon_trainable_variables
xlayer_metrics
ymetrics
 
 
 
?
zlayer_regularization_losses
4regularization_losses

{layers
5trainable_variables
6	variables
|non_trainable_variables
}layer_metrics
~metrics
 
 
 
?
layer_regularization_losses
8regularization_losses
?layers
9trainable_variables
:	variables
?non_trainable_variables
?layer_metrics
?metrics
 
 
 
?
 ?layer_regularization_losses
<regularization_losses
?layers
=trainable_variables
>	variables
?non_trainable_variables
?layer_metrics
?metrics
ZX
VARIABLE_VALUElayer10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
?
 ?layer_regularization_losses
Bregularization_losses
?layers
Ctrainable_variables
D	variables
?non_trainable_variables
?layer_metrics
?metrics
ZX
VARIABLE_VALUElayer11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer11/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
?
 ?layer_regularization_losses
Hregularization_losses
?layers
Itrainable_variables
J	variables
?non_trainable_variables
?layer_metrics
?metrics
YW
VARIABLE_VALUEsymbol/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEsymbol/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
?
 ?layer_regularization_losses
Nregularization_losses
?layers
Otrainable_variables
P	variables
?non_trainable_variables
?layer_metrics
?metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
^
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
 
 

?0
?1
?2
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
|z
VARIABLE_VALUEAdam/layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer6/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer6/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer10/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer10/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer11/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer11/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/symbol/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/symbol/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer6/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer6/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer10/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer10/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer11/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer11/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/symbol/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/symbol/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_7Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7layer1/kernellayer1/biaslayer2/kernellayer2/biaslayer5/kernellayer5/biaslayer6/kernellayer6/biaslayer10/kernellayer10/biaslayer11/kernellayer11/biassymbol/kernelsymbol/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_116873
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer5/kernel/Read/ReadVariableOplayer5/bias/Read/ReadVariableOp!layer6/kernel/Read/ReadVariableOplayer6/bias/Read/ReadVariableOp"layer10/kernel/Read/ReadVariableOp layer10/bias/Read/ReadVariableOp"layer11/kernel/Read/ReadVariableOp layer11/bias/Read/ReadVariableOp!symbol/kernel/Read/ReadVariableOpsymbol/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp(Adam/layer1/kernel/m/Read/ReadVariableOp&Adam/layer1/bias/m/Read/ReadVariableOp(Adam/layer2/kernel/m/Read/ReadVariableOp&Adam/layer2/bias/m/Read/ReadVariableOp(Adam/layer5/kernel/m/Read/ReadVariableOp&Adam/layer5/bias/m/Read/ReadVariableOp(Adam/layer6/kernel/m/Read/ReadVariableOp&Adam/layer6/bias/m/Read/ReadVariableOp)Adam/layer10/kernel/m/Read/ReadVariableOp'Adam/layer10/bias/m/Read/ReadVariableOp)Adam/layer11/kernel/m/Read/ReadVariableOp'Adam/layer11/bias/m/Read/ReadVariableOp(Adam/symbol/kernel/m/Read/ReadVariableOp&Adam/symbol/bias/m/Read/ReadVariableOp(Adam/layer1/kernel/v/Read/ReadVariableOp&Adam/layer1/bias/v/Read/ReadVariableOp(Adam/layer2/kernel/v/Read/ReadVariableOp&Adam/layer2/bias/v/Read/ReadVariableOp(Adam/layer5/kernel/v/Read/ReadVariableOp&Adam/layer5/bias/v/Read/ReadVariableOp(Adam/layer6/kernel/v/Read/ReadVariableOp&Adam/layer6/bias/v/Read/ReadVariableOp)Adam/layer10/kernel/v/Read/ReadVariableOp'Adam/layer10/bias/v/Read/ReadVariableOp)Adam/layer11/kernel/v/Read/ReadVariableOp'Adam/layer11/bias/v/Read/ReadVariableOp(Adam/symbol/kernel/v/Read/ReadVariableOp&Adam/symbol/bias/v/Read/ReadVariableOpConst*B
Tin;
927	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_117462
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer5/kernellayer5/biaslayer6/kernellayer6/biaslayer10/kernellayer10/biaslayer11/kernellayer11/biassymbol/kernelsymbol/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/layer1/kernel/mAdam/layer1/bias/mAdam/layer2/kernel/mAdam/layer2/bias/mAdam/layer5/kernel/mAdam/layer5/bias/mAdam/layer6/kernel/mAdam/layer6/bias/mAdam/layer10/kernel/mAdam/layer10/bias/mAdam/layer11/kernel/mAdam/layer11/bias/mAdam/symbol/kernel/mAdam/symbol/bias/mAdam/layer1/kernel/vAdam/layer1/bias/vAdam/layer2/kernel/vAdam/layer2/bias/vAdam/layer5/kernel/vAdam/layer5/bias/vAdam/layer6/kernel/vAdam/layer6/bias/vAdam/layer10/kernel/vAdam/layer10/bias/vAdam/layer11/kernel/vAdam/layer11/bias/vAdam/symbol/kernel/vAdam/symbol/bias/v*A
Tin:
826*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_117631̜	
?
a
B__inference_layer8_layer_call_and_return_conditional_losses_117194

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0*

seed{2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_117631
file_prefix"
assignvariableop_layer1_kernel"
assignvariableop_1_layer1_bias$
 assignvariableop_2_layer2_kernel"
assignvariableop_3_layer2_bias$
 assignvariableop_4_layer5_kernel"
assignvariableop_5_layer5_bias$
 assignvariableop_6_layer6_kernel"
assignvariableop_7_layer6_bias%
!assignvariableop_8_layer10_kernel#
assignvariableop_9_layer10_bias&
"assignvariableop_10_layer11_kernel$
 assignvariableop_11_layer11_bias%
!assignvariableop_12_symbol_kernel#
assignvariableop_13_symbol_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate
assignvariableop_19_total
assignvariableop_20_count
assignvariableop_21_total_1
assignvariableop_22_count_1
assignvariableop_23_total_2
assignvariableop_24_count_2,
(assignvariableop_25_adam_layer1_kernel_m*
&assignvariableop_26_adam_layer1_bias_m,
(assignvariableop_27_adam_layer2_kernel_m*
&assignvariableop_28_adam_layer2_bias_m,
(assignvariableop_29_adam_layer5_kernel_m*
&assignvariableop_30_adam_layer5_bias_m,
(assignvariableop_31_adam_layer6_kernel_m*
&assignvariableop_32_adam_layer6_bias_m-
)assignvariableop_33_adam_layer10_kernel_m+
'assignvariableop_34_adam_layer10_bias_m-
)assignvariableop_35_adam_layer11_kernel_m+
'assignvariableop_36_adam_layer11_bias_m,
(assignvariableop_37_adam_symbol_kernel_m*
&assignvariableop_38_adam_symbol_bias_m,
(assignvariableop_39_adam_layer1_kernel_v*
&assignvariableop_40_adam_layer1_bias_v,
(assignvariableop_41_adam_layer2_kernel_v*
&assignvariableop_42_adam_layer2_bias_v,
(assignvariableop_43_adam_layer5_kernel_v*
&assignvariableop_44_adam_layer5_bias_v,
(assignvariableop_45_adam_layer6_kernel_v*
&assignvariableop_46_adam_layer6_bias_v-
)assignvariableop_47_adam_layer10_kernel_v+
'assignvariableop_48_adam_layer10_bias_v-
)assignvariableop_49_adam_layer11_kernel_v+
'assignvariableop_50_adam_layer11_bias_v,
(assignvariableop_51_adam_symbol_kernel_v*
&assignvariableop_52_adam_symbol_bias_v
identity_54??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_layer10_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer10_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_layer11_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_layer11_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_symbol_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_symbol_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_layer1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_layer1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_layer2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_layer2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_layer5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_layer5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_layer6_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_layer6_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_layer10_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_layer10_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_layer11_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_layer11_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_symbol_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_symbol_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_layer1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_layer1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_layer2_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_layer2_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_layer5_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_layer5_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_layer6_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_layer6_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_layer10_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_layer10_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_layer11_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_layer11_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_symbol_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp&assignvariableop_52_adam_symbol_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53?	
Identity_54IdentityIdentity_53:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_54"#
identity_54Identity_54:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
`
B__inference_layer4_layer_call_and_return_conditional_losses_117132

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????$$2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????$$2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????$$:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?
C
'__inference_layer9_layer_call_fn_117220

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer9_layer_call_and_return_conditional_losses_1165352
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
B__inference_layer4_layer_call_and_return_conditional_losses_116430

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????$$2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????$$2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????$$:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?
C
'__inference_layer7_layer_call_fn_116353

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
B__inference_layer6_layer_call_and_return_conditional_losses_116481

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????!! ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????!! 
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_116873
input_7
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1163292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?	
?
&__inference_Logan_layer_call_fn_116830
input_7
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Logan_layer_call_and_return_conditional_losses_1167992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?
|
'__inference_layer1_layer_call_fn_117095

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_1163682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_layer6_layer_call_and_return_conditional_losses_117173

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????!! ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????!! 
 
_user_specified_nameinputs
?
C
'__inference_layer4_layer_call_fn_117142

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_1164302
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????$$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$$:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?
^
B__inference_layer9_layer_call_and_return_conditional_losses_116535

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
|
'__inference_symbol_layer_call_fn_117280

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_symbol_layer_call_and_return_conditional_losses_1166082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_layer3_layer_call_and_return_conditional_losses_116335

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
B__inference_layer5_layer_call_and_return_conditional_losses_117153

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!! 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????!! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?;
?
A__inference_Logan_layer_call_and_return_conditional_losses_116625
input_7
layer1_116379
layer1_116381
layer2_116406
layer2_116408
layer5_116465
layer5_116467
layer6_116492
layer6_116494
layer10_116565
layer10_116567
layer11_116592
layer11_116594
symbol_116619
symbol_116621
identity??layer1/StatefulPartitionedCall?layer10/StatefulPartitionedCall?layer11/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer4/StatefulPartitionedCall?layer5/StatefulPartitionedCall?layer6/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_7layer1_116379layer1_116381*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_1163682 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_116406layer2_116408*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_1163952 
layer2/StatefulPartitionedCall?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
layer3/PartitionedCall?
layer3/PartitionedCall_1PartitionedCalllayer3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
layer3/PartitionedCall_1?
layer4/StatefulPartitionedCallStatefulPartitionedCall!layer3/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_1164252 
layer4/StatefulPartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_116465layer5_116467*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_1164542 
layer5/StatefulPartitionedCall?
layer6/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0layer6_116492layer6_116494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_1164812 
layer6/StatefulPartitionedCall?
layer7/PartitionedCallPartitionedCall'layer6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
layer7/PartitionedCall?
layer7/PartitionedCall_1PartitionedCalllayer7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
layer7/PartitionedCall_1?
layer8/StatefulPartitionedCallStatefulPartitionedCall!layer7/PartitionedCall_1:output:0^layer4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_1165112 
layer8/StatefulPartitionedCall?
layer9/PartitionedCallPartitionedCall'layer8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer9_layer_call_and_return_conditional_losses_1165352
layer9/PartitionedCall?
layer10/StatefulPartitionedCallStatefulPartitionedCalllayer9/PartitionedCall:output:0layer10_116565layer10_116567*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer10_layer_call_and_return_conditional_losses_1165542!
layer10/StatefulPartitionedCall?
layer11/StatefulPartitionedCallStatefulPartitionedCall(layer10/StatefulPartitionedCall:output:0layer11_116592layer11_116594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer11_layer_call_and_return_conditional_losses_1165812!
layer11/StatefulPartitionedCall?
symbol/StatefulPartitionedCallStatefulPartitionedCall(layer11/StatefulPartitionedCall:output:0symbol_116619symbol_116621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_symbol_layer_call_and_return_conditional_losses_1166082 
symbol/StatefulPartitionedCall?
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall ^layer10/StatefulPartitionedCall ^layer11/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer6/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2B
layer10/StatefulPartitionedCalllayer10/StatefulPartitionedCall2B
layer11/StatefulPartitionedCalllayer11/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?

?
B__inference_layer2_layer_call_and_return_conditional_losses_117106

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
C__inference_layer11_layer_call_and_return_conditional_losses_116581

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_layer10_layer_call_and_return_conditional_losses_116554

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_layer8_layer_call_and_return_conditional_losses_117199

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
}
(__inference_layer11_layer_call_fn_117260

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer11_layer_call_and_return_conditional_losses_1165812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
B__inference_layer8_layer_call_and_return_conditional_losses_116511

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0*

seed{2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?L
?	
A__inference_Logan_layer_call_and_return_conditional_losses_117009

inputs)
%layer1_conv2d_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_conv2d_readvariableop_resource*
&layer2_biasadd_readvariableop_resource)
%layer5_conv2d_readvariableop_resource*
&layer5_biasadd_readvariableop_resource)
%layer6_conv2d_readvariableop_resource*
&layer6_biasadd_readvariableop_resource*
&layer10_matmul_readvariableop_resource+
'layer10_biasadd_readvariableop_resource*
&layer11_matmul_readvariableop_resource+
'layer11_biasadd_readvariableop_resource)
%symbol_matmul_readvariableop_resource*
&symbol_biasadd_readvariableop_resource
identity??layer1/BiasAdd/ReadVariableOp?layer1/Conv2D/ReadVariableOp?layer10/BiasAdd/ReadVariableOp?layer10/MatMul/ReadVariableOp?layer11/BiasAdd/ReadVariableOp?layer11/MatMul/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/Conv2D/ReadVariableOp?layer5/BiasAdd/ReadVariableOp?layer5/Conv2D/ReadVariableOp?layer6/BiasAdd/ReadVariableOp?layer6/Conv2D/ReadVariableOp?symbol/BiasAdd/ReadVariableOp?symbol/MatMul/ReadVariableOp?
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer1/Conv2D/ReadVariableOp?
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
layer1/Conv2D?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
layer1/BiasAddw
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
layer1/Relu?
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer2/Conv2D/ReadVariableOp?
layer2/Conv2DConv2Dlayer1/Relu:activations:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
layer2/Conv2D?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
layer2/BiasAddw
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
layer2/Relu?
layer3/MaxPoolMaxPoollayer2/Relu:activations:0*/
_output_shapes
:?????????HH*
ksize
*
paddingVALID*
strides
2
layer3/MaxPool?
layer3/MaxPool_1MaxPoollayer3/MaxPool:output:0*/
_output_shapes
:?????????$$*
ksize
*
paddingVALID*
strides
2
layer3/MaxPool_1?
layer4/IdentityIdentitylayer3/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????$$2
layer4/Identity?
layer5/Conv2D/ReadVariableOpReadVariableOp%layer5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
layer5/Conv2D/ReadVariableOp?
layer5/Conv2DConv2Dlayer4/Identity:output:0$layer5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! *
paddingVALID*
strides
2
layer5/Conv2D?
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer5/BiasAdd/ReadVariableOp?
layer5/BiasAddBiasAddlayer5/Conv2D:output:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! 2
layer5/BiasAddu
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!! 2
layer5/Relu?
layer6/Conv2D/ReadVariableOpReadVariableOp%layer6_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
layer6/Conv2D/ReadVariableOp?
layer6/Conv2DConv2Dlayer5/Relu:activations:0$layer6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
layer6/Conv2D?
layer6/BiasAdd/ReadVariableOpReadVariableOp&layer6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer6/BiasAdd/ReadVariableOp?
layer6/BiasAddBiasAddlayer6/Conv2D:output:0%layer6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
layer6/BiasAddu
layer6/ReluRelulayer6/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
layer6/Relu?
layer7/MaxPoolMaxPoollayer6/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
layer7/MaxPool?
layer7/MaxPool_1MaxPoollayer7/MaxPool:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
layer7/MaxPool_1?
layer8/IdentityIdentitylayer7/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
layer8/Identitym
layer9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
layer9/Const?
layer9/ReshapeReshapelayer8/Identity:output:0layer9/Const:output:0*
T0*(
_output_shapes
:??????????2
layer9/Reshape?
layer10/MatMul/ReadVariableOpReadVariableOp&layer10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer10/MatMul/ReadVariableOp?
layer10/MatMulMatMullayer9/Reshape:output:0%layer10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer10/MatMul?
layer10/BiasAdd/ReadVariableOpReadVariableOp'layer10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer10/BiasAdd/ReadVariableOp?
layer10/BiasAddBiasAddlayer10/MatMul:product:0&layer10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer10/BiasAddq
layer10/ReluRelulayer10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer10/Relu?
layer11/MatMul/ReadVariableOpReadVariableOp&layer11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer11/MatMul/ReadVariableOp?
layer11/MatMulMatMullayer10/Relu:activations:0%layer11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer11/MatMul?
layer11/BiasAdd/ReadVariableOpReadVariableOp'layer11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer11/BiasAdd/ReadVariableOp?
layer11/BiasAddBiasAddlayer11/MatMul:product:0&layer11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer11/BiasAddq
layer11/ReluRelulayer11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer11/Relu?
symbol/MatMul/ReadVariableOpReadVariableOp%symbol_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
symbol/MatMul/ReadVariableOp?
symbol/MatMulMatMullayer11/Relu:activations:0$symbol/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
symbol/MatMul?
symbol/BiasAdd/ReadVariableOpReadVariableOp&symbol_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
symbol/BiasAdd/ReadVariableOp?
symbol/BiasAddBiasAddsymbol/MatMul:product:0%symbol/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
symbol/BiasAddv
symbol/SoftmaxSoftmaxsymbol/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
symbol/Softmax?
IdentityIdentitysymbol/Softmax:softmax:0^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer10/BiasAdd/ReadVariableOp^layer10/MatMul/ReadVariableOp^layer11/BiasAdd/ReadVariableOp^layer11/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/Conv2D/ReadVariableOp^layer6/BiasAdd/ReadVariableOp^layer6/Conv2D/ReadVariableOp^symbol/BiasAdd/ReadVariableOp^symbol/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2@
layer10/BiasAdd/ReadVariableOplayer10/BiasAdd/ReadVariableOp2>
layer10/MatMul/ReadVariableOplayer10/MatMul/ReadVariableOp2@
layer11/BiasAdd/ReadVariableOplayer11/BiasAdd/ReadVariableOp2>
layer11/MatMul/ReadVariableOplayer11/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/Conv2D/ReadVariableOplayer5/Conv2D/ReadVariableOp2>
layer6/BiasAdd/ReadVariableOplayer6/BiasAdd/ReadVariableOp2<
layer6/Conv2D/ReadVariableOplayer6/Conv2D/ReadVariableOp2>
symbol/BiasAdd/ReadVariableOpsymbol/BiasAdd/ReadVariableOp2<
symbol/MatMul/ReadVariableOpsymbol/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
|
'__inference_layer5_layer_call_fn_117162

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_1164542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????!! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?	
?
&__inference_Logan_layer_call_fn_117042

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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Logan_layer_call_and_return_conditional_losses_1167202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?8
?
A__inference_Logan_layer_call_and_return_conditional_losses_116671
input_7
layer1_116628
layer1_116630
layer2_116633
layer2_116635
layer5_116641
layer5_116643
layer6_116646
layer6_116648
layer10_116655
layer10_116657
layer11_116660
layer11_116662
symbol_116665
symbol_116667
identity??layer1/StatefulPartitionedCall?layer10/StatefulPartitionedCall?layer11/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer5/StatefulPartitionedCall?layer6/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_7layer1_116628layer1_116630*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_1163682 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_116633layer2_116635*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_1163952 
layer2/StatefulPartitionedCall?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
layer3/PartitionedCall?
layer3/PartitionedCall_1PartitionedCalllayer3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
layer3/PartitionedCall_1?
layer4/PartitionedCallPartitionedCall!layer3/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_1164302
layer4/PartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCalllayer4/PartitionedCall:output:0layer5_116641layer5_116643*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_1164542 
layer5/StatefulPartitionedCall?
layer6/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0layer6_116646layer6_116648*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_1164812 
layer6/StatefulPartitionedCall?
layer7/PartitionedCallPartitionedCall'layer6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
layer7/PartitionedCall?
layer7/PartitionedCall_1PartitionedCalllayer7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
layer7/PartitionedCall_1?
layer8/PartitionedCallPartitionedCall!layer7/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_1165162
layer8/PartitionedCall?
layer9/PartitionedCallPartitionedCalllayer8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer9_layer_call_and_return_conditional_losses_1165352
layer9/PartitionedCall?
layer10/StatefulPartitionedCallStatefulPartitionedCalllayer9/PartitionedCall:output:0layer10_116655layer10_116657*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer10_layer_call_and_return_conditional_losses_1165542!
layer10/StatefulPartitionedCall?
layer11/StatefulPartitionedCallStatefulPartitionedCall(layer10/StatefulPartitionedCall:output:0layer11_116660layer11_116662*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer11_layer_call_and_return_conditional_losses_1165812!
layer11/StatefulPartitionedCall?
symbol/StatefulPartitionedCallStatefulPartitionedCall(layer11/StatefulPartitionedCall:output:0symbol_116665symbol_116667*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_symbol_layer_call_and_return_conditional_losses_1166082 
symbol/StatefulPartitionedCall?
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall ^layer10/StatefulPartitionedCall ^layer11/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer6/StatefulPartitionedCall^symbol/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2B
layer10/StatefulPartitionedCalllayer10/StatefulPartitionedCall2B
layer11/StatefulPartitionedCalllayer11/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?i
?
__inference__traced_save_117462
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer5_kernel_read_readvariableop*
&savev2_layer5_bias_read_readvariableop,
(savev2_layer6_kernel_read_readvariableop*
&savev2_layer6_bias_read_readvariableop-
)savev2_layer10_kernel_read_readvariableop+
'savev2_layer10_bias_read_readvariableop-
)savev2_layer11_kernel_read_readvariableop+
'savev2_layer11_bias_read_readvariableop,
(savev2_symbol_kernel_read_readvariableop*
&savev2_symbol_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop3
/savev2_adam_layer5_kernel_m_read_readvariableop1
-savev2_adam_layer5_bias_m_read_readvariableop3
/savev2_adam_layer6_kernel_m_read_readvariableop1
-savev2_adam_layer6_bias_m_read_readvariableop4
0savev2_adam_layer10_kernel_m_read_readvariableop2
.savev2_adam_layer10_bias_m_read_readvariableop4
0savev2_adam_layer11_kernel_m_read_readvariableop2
.savev2_adam_layer11_bias_m_read_readvariableop3
/savev2_adam_symbol_kernel_m_read_readvariableop1
-savev2_adam_symbol_bias_m_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop3
/savev2_adam_layer5_kernel_v_read_readvariableop1
-savev2_adam_layer5_bias_v_read_readvariableop3
/savev2_adam_layer6_kernel_v_read_readvariableop1
-savev2_adam_layer6_bias_v_read_readvariableop4
0savev2_adam_layer10_kernel_v_read_readvariableop2
.savev2_adam_layer10_bias_v_read_readvariableop4
0savev2_adam_layer11_kernel_v_read_readvariableop2
.savev2_adam_layer11_bias_v_read_readvariableop3
/savev2_adam_symbol_kernel_v_read_readvariableop1
-savev2_adam_symbol_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*?
value?B?6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer5_kernel_read_readvariableop&savev2_layer5_bias_read_readvariableop(savev2_layer6_kernel_read_readvariableop&savev2_layer6_bias_read_readvariableop)savev2_layer10_kernel_read_readvariableop'savev2_layer10_bias_read_readvariableop)savev2_layer11_kernel_read_readvariableop'savev2_layer11_bias_read_readvariableop(savev2_symbol_kernel_read_readvariableop&savev2_symbol_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_layer5_kernel_m_read_readvariableop-savev2_adam_layer5_bias_m_read_readvariableop/savev2_adam_layer6_kernel_m_read_readvariableop-savev2_adam_layer6_bias_m_read_readvariableop0savev2_adam_layer10_kernel_m_read_readvariableop.savev2_adam_layer10_bias_m_read_readvariableop0savev2_adam_layer11_kernel_m_read_readvariableop.savev2_adam_layer11_bias_m_read_readvariableop/savev2_adam_symbol_kernel_m_read_readvariableop-savev2_adam_symbol_bias_m_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_layer5_kernel_v_read_readvariableop-savev2_adam_layer5_bias_v_read_readvariableop/savev2_adam_layer6_kernel_v_read_readvariableop-savev2_adam_layer6_bias_v_read_readvariableop0savev2_adam_layer10_kernel_v_read_readvariableop.savev2_adam_layer10_bias_v_read_readvariableop0savev2_adam_layer11_kernel_v_read_readvariableop.savev2_adam_layer11_bias_v_read_readvariableop/savev2_adam_symbol_kernel_v_read_readvariableop-savev2_adam_symbol_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : :  : :
??:?:
??:?:	?:: : : : : : : : : : : ::::: : :  : :
??:?:
??:?:	?:::::: : :  : :
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
:  : !

_output_shapes
: :&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: :,.(
&
_output_shapes
:  : /

_output_shapes
: :&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:&2"
 
_output_shapes
:
??:!3

_output_shapes	
:?:%4!

_output_shapes
:	?: 5

_output_shapes
::6

_output_shapes
: 
?
`
'__inference_layer4_layer_call_fn_117137

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_1164252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????$$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$$22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?^
?	
A__inference_Logan_layer_call_and_return_conditional_losses_116948

inputs)
%layer1_conv2d_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_conv2d_readvariableop_resource*
&layer2_biasadd_readvariableop_resource)
%layer5_conv2d_readvariableop_resource*
&layer5_biasadd_readvariableop_resource)
%layer6_conv2d_readvariableop_resource*
&layer6_biasadd_readvariableop_resource*
&layer10_matmul_readvariableop_resource+
'layer10_biasadd_readvariableop_resource*
&layer11_matmul_readvariableop_resource+
'layer11_biasadd_readvariableop_resource)
%symbol_matmul_readvariableop_resource*
&symbol_biasadd_readvariableop_resource
identity??layer1/BiasAdd/ReadVariableOp?layer1/Conv2D/ReadVariableOp?layer10/BiasAdd/ReadVariableOp?layer10/MatMul/ReadVariableOp?layer11/BiasAdd/ReadVariableOp?layer11/MatMul/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/Conv2D/ReadVariableOp?layer5/BiasAdd/ReadVariableOp?layer5/Conv2D/ReadVariableOp?layer6/BiasAdd/ReadVariableOp?layer6/Conv2D/ReadVariableOp?symbol/BiasAdd/ReadVariableOp?symbol/MatMul/ReadVariableOp?
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer1/Conv2D/ReadVariableOp?
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
layer1/Conv2D?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer1/BiasAdd/ReadVariableOp?
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
layer1/BiasAddw
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
layer1/Relu?
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer2/Conv2D/ReadVariableOp?
layer2/Conv2DConv2Dlayer1/Relu:activations:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
layer2/Conv2D?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer2/BiasAdd/ReadVariableOp?
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
layer2/BiasAddw
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
layer2/Relu?
layer3/MaxPoolMaxPoollayer2/Relu:activations:0*/
_output_shapes
:?????????HH*
ksize
*
paddingVALID*
strides
2
layer3/MaxPool?
layer3/MaxPool_1MaxPoollayer3/MaxPool:output:0*/
_output_shapes
:?????????$$*
ksize
*
paddingVALID*
strides
2
layer3/MaxPool_1q
layer4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer4/dropout/Const?
layer4/dropout/MulMullayer3/MaxPool_1:output:0layer4/dropout/Const:output:0*
T0*/
_output_shapes
:?????????$$2
layer4/dropout/Mulu
layer4/dropout/ShapeShapelayer3/MaxPool_1:output:0*
T0*
_output_shapes
:2
layer4/dropout/Shape?
+layer4/dropout/random_uniform/RandomUniformRandomUniformlayer4/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????$$*
dtype0*

seed{2-
+layer4/dropout/random_uniform/RandomUniform?
layer4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer4/dropout/GreaterEqual/y?
layer4/dropout/GreaterEqualGreaterEqual4layer4/dropout/random_uniform/RandomUniform:output:0&layer4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????$$2
layer4/dropout/GreaterEqual?
layer4/dropout/CastCastlayer4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????$$2
layer4/dropout/Cast?
layer4/dropout/Mul_1Mullayer4/dropout/Mul:z:0layer4/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????$$2
layer4/dropout/Mul_1?
layer5/Conv2D/ReadVariableOpReadVariableOp%layer5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
layer5/Conv2D/ReadVariableOp?
layer5/Conv2DConv2Dlayer4/dropout/Mul_1:z:0$layer5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! *
paddingVALID*
strides
2
layer5/Conv2D?
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer5/BiasAdd/ReadVariableOp?
layer5/BiasAddBiasAddlayer5/Conv2D:output:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! 2
layer5/BiasAddu
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!! 2
layer5/Relu?
layer6/Conv2D/ReadVariableOpReadVariableOp%layer6_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
layer6/Conv2D/ReadVariableOp?
layer6/Conv2DConv2Dlayer5/Relu:activations:0$layer6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
layer6/Conv2D?
layer6/BiasAdd/ReadVariableOpReadVariableOp&layer6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer6/BiasAdd/ReadVariableOp?
layer6/BiasAddBiasAddlayer6/Conv2D:output:0%layer6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
layer6/BiasAddu
layer6/ReluRelulayer6/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
layer6/Relu?
layer7/MaxPoolMaxPoollayer6/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
layer7/MaxPool?
layer7/MaxPool_1MaxPoollayer7/MaxPool:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
layer7/MaxPool_1q
layer8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer8/dropout/Const?
layer8/dropout/MulMullayer7/MaxPool_1:output:0layer8/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
layer8/dropout/Mulu
layer8/dropout/ShapeShapelayer7/MaxPool_1:output:0*
T0*
_output_shapes
:2
layer8/dropout/Shape?
+layer8/dropout/random_uniform/RandomUniformRandomUniformlayer8/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0*

seed{*
seed22-
+layer8/dropout/random_uniform/RandomUniform?
layer8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer8/dropout/GreaterEqual/y?
layer8/dropout/GreaterEqualGreaterEqual4layer8/dropout/random_uniform/RandomUniform:output:0&layer8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
layer8/dropout/GreaterEqual?
layer8/dropout/CastCastlayer8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
layer8/dropout/Cast?
layer8/dropout/Mul_1Mullayer8/dropout/Mul:z:0layer8/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
layer8/dropout/Mul_1m
layer9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
layer9/Const?
layer9/ReshapeReshapelayer8/dropout/Mul_1:z:0layer9/Const:output:0*
T0*(
_output_shapes
:??????????2
layer9/Reshape?
layer10/MatMul/ReadVariableOpReadVariableOp&layer10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer10/MatMul/ReadVariableOp?
layer10/MatMulMatMullayer9/Reshape:output:0%layer10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer10/MatMul?
layer10/BiasAdd/ReadVariableOpReadVariableOp'layer10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer10/BiasAdd/ReadVariableOp?
layer10/BiasAddBiasAddlayer10/MatMul:product:0&layer10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer10/BiasAddq
layer10/ReluRelulayer10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer10/Relu?
layer11/MatMul/ReadVariableOpReadVariableOp&layer11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer11/MatMul/ReadVariableOp?
layer11/MatMulMatMullayer10/Relu:activations:0%layer11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer11/MatMul?
layer11/BiasAdd/ReadVariableOpReadVariableOp'layer11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer11/BiasAdd/ReadVariableOp?
layer11/BiasAddBiasAddlayer11/MatMul:product:0&layer11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer11/BiasAddq
layer11/ReluRelulayer11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer11/Relu?
symbol/MatMul/ReadVariableOpReadVariableOp%symbol_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
symbol/MatMul/ReadVariableOp?
symbol/MatMulMatMullayer11/Relu:activations:0$symbol/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
symbol/MatMul?
symbol/BiasAdd/ReadVariableOpReadVariableOp&symbol_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
symbol/BiasAdd/ReadVariableOp?
symbol/BiasAddBiasAddsymbol/MatMul:product:0%symbol/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
symbol/BiasAddv
symbol/SoftmaxSoftmaxsymbol/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
symbol/Softmax?
IdentityIdentitysymbol/Softmax:softmax:0^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer10/BiasAdd/ReadVariableOp^layer10/MatMul/ReadVariableOp^layer11/BiasAdd/ReadVariableOp^layer11/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/Conv2D/ReadVariableOp^layer6/BiasAdd/ReadVariableOp^layer6/Conv2D/ReadVariableOp^symbol/BiasAdd/ReadVariableOp^symbol/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2@
layer10/BiasAdd/ReadVariableOplayer10/BiasAdd/ReadVariableOp2>
layer10/MatMul/ReadVariableOplayer10/MatMul/ReadVariableOp2@
layer11/BiasAdd/ReadVariableOplayer11/BiasAdd/ReadVariableOp2>
layer11/MatMul/ReadVariableOplayer11/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/Conv2D/ReadVariableOplayer5/Conv2D/ReadVariableOp2>
layer6/BiasAdd/ReadVariableOplayer6/BiasAdd/ReadVariableOp2<
layer6/Conv2D/ReadVariableOplayer6/Conv2D/ReadVariableOp2>
symbol/BiasAdd/ReadVariableOpsymbol/BiasAdd/ReadVariableOp2<
symbol/MatMul/ReadVariableOpsymbol/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?;
?
A__inference_Logan_layer_call_and_return_conditional_losses_116720

inputs
layer1_116677
layer1_116679
layer2_116682
layer2_116684
layer5_116690
layer5_116692
layer6_116695
layer6_116697
layer10_116704
layer10_116706
layer11_116709
layer11_116711
symbol_116714
symbol_116716
identity??layer1/StatefulPartitionedCall?layer10/StatefulPartitionedCall?layer11/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer4/StatefulPartitionedCall?layer5/StatefulPartitionedCall?layer6/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_116677layer1_116679*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_1163682 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_116682layer2_116684*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_1163952 
layer2/StatefulPartitionedCall?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
layer3/PartitionedCall?
layer3/PartitionedCall_1PartitionedCalllayer3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
layer3/PartitionedCall_1?
layer4/StatefulPartitionedCallStatefulPartitionedCall!layer3/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_1164252 
layer4/StatefulPartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_116690layer5_116692*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_1164542 
layer5/StatefulPartitionedCall?
layer6/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0layer6_116695layer6_116697*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_1164812 
layer6/StatefulPartitionedCall?
layer7/PartitionedCallPartitionedCall'layer6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
layer7/PartitionedCall?
layer7/PartitionedCall_1PartitionedCalllayer7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
layer7/PartitionedCall_1?
layer8/StatefulPartitionedCallStatefulPartitionedCall!layer7/PartitionedCall_1:output:0^layer4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_1165112 
layer8/StatefulPartitionedCall?
layer9/PartitionedCallPartitionedCall'layer8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer9_layer_call_and_return_conditional_losses_1165352
layer9/PartitionedCall?
layer10/StatefulPartitionedCallStatefulPartitionedCalllayer9/PartitionedCall:output:0layer10_116704layer10_116706*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer10_layer_call_and_return_conditional_losses_1165542!
layer10/StatefulPartitionedCall?
layer11/StatefulPartitionedCallStatefulPartitionedCall(layer10/StatefulPartitionedCall:output:0layer11_116709layer11_116711*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer11_layer_call_and_return_conditional_losses_1165812!
layer11/StatefulPartitionedCall?
symbol/StatefulPartitionedCallStatefulPartitionedCall(layer11/StatefulPartitionedCall:output:0symbol_116714symbol_116716*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_symbol_layer_call_and_return_conditional_losses_1166082 
symbol/StatefulPartitionedCall?
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall ^layer10/StatefulPartitionedCall ^layer11/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer6/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2B
layer10/StatefulPartitionedCalllayer10/StatefulPartitionedCall2B
layer11/StatefulPartitionedCalllayer11/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_layer1_layer_call_and_return_conditional_losses_117086

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
|
'__inference_layer2_layer_call_fn_117115

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_1163952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
&__inference_Logan_layer_call_fn_117075

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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Logan_layer_call_and_return_conditional_losses_1167992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
'__inference_layer8_layer_call_fn_117204

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_1165112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
&__inference_Logan_layer_call_fn_116751
input_7
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

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Logan_layer_call_and_return_conditional_losses_1167202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?	
?
B__inference_symbol_layer_call_and_return_conditional_losses_117271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_layer5_layer_call_and_return_conditional_losses_116454

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????!! 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????!! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????$$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?
a
B__inference_layer4_layer_call_and_return_conditional_losses_116425

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????$$2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????$$*
dtype0*

seed{2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????$$2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????$$2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????$$2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????$$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$$:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?
}
(__inference_layer10_layer_call_fn_117240

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer10_layer_call_and_return_conditional_losses_1165542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_layer9_layer_call_and_return_conditional_losses_117215

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
B__inference_symbol_layer_call_and_return_conditional_losses_116608

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_layer7_layer_call_and_return_conditional_losses_116347

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?8
?
A__inference_Logan_layer_call_and_return_conditional_losses_116799

inputs
layer1_116756
layer1_116758
layer2_116761
layer2_116763
layer5_116769
layer5_116771
layer6_116774
layer6_116776
layer10_116783
layer10_116785
layer11_116788
layer11_116790
symbol_116793
symbol_116795
identity??layer1/StatefulPartitionedCall?layer10/StatefulPartitionedCall?layer11/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer5/StatefulPartitionedCall?layer6/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_116756layer1_116758*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_1163682 
layer1/StatefulPartitionedCall?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_116761layer2_116763*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_1163952 
layer2/StatefulPartitionedCall?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????HH* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
layer3/PartitionedCall?
layer3/PartitionedCall_1PartitionedCalllayer3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
layer3/PartitionedCall_1?
layer4/PartitionedCallPartitionedCall!layer3/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer4_layer_call_and_return_conditional_losses_1164302
layer4/PartitionedCall?
layer5/StatefulPartitionedCallStatefulPartitionedCalllayer4/PartitionedCall:output:0layer5_116769layer5_116771*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????!! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer5_layer_call_and_return_conditional_losses_1164542 
layer5/StatefulPartitionedCall?
layer6/StatefulPartitionedCallStatefulPartitionedCall'layer5/StatefulPartitionedCall:output:0layer6_116774layer6_116776*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_1164812 
layer6/StatefulPartitionedCall?
layer7/PartitionedCallPartitionedCall'layer6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
layer7/PartitionedCall?
layer7/PartitionedCall_1PartitionedCalllayer7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_1163472
layer7/PartitionedCall_1?
layer8/PartitionedCallPartitionedCall!layer7/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_1165162
layer8/PartitionedCall?
layer9/PartitionedCallPartitionedCalllayer8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer9_layer_call_and_return_conditional_losses_1165352
layer9/PartitionedCall?
layer10/StatefulPartitionedCallStatefulPartitionedCalllayer9/PartitionedCall:output:0layer10_116783layer10_116785*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer10_layer_call_and_return_conditional_losses_1165542!
layer10/StatefulPartitionedCall?
layer11/StatefulPartitionedCallStatefulPartitionedCall(layer10/StatefulPartitionedCall:output:0layer11_116788layer11_116790*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_layer11_layer_call_and_return_conditional_losses_1165812!
layer11/StatefulPartitionedCall?
symbol/StatefulPartitionedCallStatefulPartitionedCall(layer11/StatefulPartitionedCall:output:0symbol_116793symbol_116795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_symbol_layer_call_and_return_conditional_losses_1166082 
symbol/StatefulPartitionedCall?
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall ^layer10/StatefulPartitionedCall ^layer11/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer6/StatefulPartitionedCall^symbol/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2B
layer10/StatefulPartitionedCalllayer10/StatefulPartitionedCall2B
layer11/StatefulPartitionedCalllayer11/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?U
?

!__inference__wrapped_model_116329
input_7/
+logan_layer1_conv2d_readvariableop_resource0
,logan_layer1_biasadd_readvariableop_resource/
+logan_layer2_conv2d_readvariableop_resource0
,logan_layer2_biasadd_readvariableop_resource/
+logan_layer5_conv2d_readvariableop_resource0
,logan_layer5_biasadd_readvariableop_resource/
+logan_layer6_conv2d_readvariableop_resource0
,logan_layer6_biasadd_readvariableop_resource0
,logan_layer10_matmul_readvariableop_resource1
-logan_layer10_biasadd_readvariableop_resource0
,logan_layer11_matmul_readvariableop_resource1
-logan_layer11_biasadd_readvariableop_resource/
+logan_symbol_matmul_readvariableop_resource0
,logan_symbol_biasadd_readvariableop_resource
identity??#Logan/layer1/BiasAdd/ReadVariableOp?"Logan/layer1/Conv2D/ReadVariableOp?$Logan/layer10/BiasAdd/ReadVariableOp?#Logan/layer10/MatMul/ReadVariableOp?$Logan/layer11/BiasAdd/ReadVariableOp?#Logan/layer11/MatMul/ReadVariableOp?#Logan/layer2/BiasAdd/ReadVariableOp?"Logan/layer2/Conv2D/ReadVariableOp?#Logan/layer5/BiasAdd/ReadVariableOp?"Logan/layer5/Conv2D/ReadVariableOp?#Logan/layer6/BiasAdd/ReadVariableOp?"Logan/layer6/Conv2D/ReadVariableOp?#Logan/symbol/BiasAdd/ReadVariableOp?"Logan/symbol/MatMul/ReadVariableOp?
"Logan/layer1/Conv2D/ReadVariableOpReadVariableOp+logan_layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"Logan/layer1/Conv2D/ReadVariableOp?
Logan/layer1/Conv2DConv2Dinput_7*Logan/layer1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Logan/layer1/Conv2D?
#Logan/layer1/BiasAdd/ReadVariableOpReadVariableOp,logan_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Logan/layer1/BiasAdd/ReadVariableOp?
Logan/layer1/BiasAddBiasAddLogan/layer1/Conv2D:output:0+Logan/layer1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
Logan/layer1/BiasAdd?
Logan/layer1/ReluReluLogan/layer1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Logan/layer1/Relu?
"Logan/layer2/Conv2D/ReadVariableOpReadVariableOp+logan_layer2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"Logan/layer2/Conv2D/ReadVariableOp?
Logan/layer2/Conv2DConv2DLogan/layer1/Relu:activations:0*Logan/layer2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Logan/layer2/Conv2D?
#Logan/layer2/BiasAdd/ReadVariableOpReadVariableOp,logan_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Logan/layer2/BiasAdd/ReadVariableOp?
Logan/layer2/BiasAddBiasAddLogan/layer2/Conv2D:output:0+Logan/layer2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
Logan/layer2/BiasAdd?
Logan/layer2/ReluReluLogan/layer2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Logan/layer2/Relu?
Logan/layer3/MaxPoolMaxPoolLogan/layer2/Relu:activations:0*/
_output_shapes
:?????????HH*
ksize
*
paddingVALID*
strides
2
Logan/layer3/MaxPool?
Logan/layer3/MaxPool_1MaxPoolLogan/layer3/MaxPool:output:0*/
_output_shapes
:?????????$$*
ksize
*
paddingVALID*
strides
2
Logan/layer3/MaxPool_1?
Logan/layer4/IdentityIdentityLogan/layer3/MaxPool_1:output:0*
T0*/
_output_shapes
:?????????$$2
Logan/layer4/Identity?
"Logan/layer5/Conv2D/ReadVariableOpReadVariableOp+logan_layer5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"Logan/layer5/Conv2D/ReadVariableOp?
Logan/layer5/Conv2DConv2DLogan/layer4/Identity:output:0*Logan/layer5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! *
paddingVALID*
strides
2
Logan/layer5/Conv2D?
#Logan/layer5/BiasAdd/ReadVariableOpReadVariableOp,logan_layer5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#Logan/layer5/BiasAdd/ReadVariableOp?
Logan/layer5/BiasAddBiasAddLogan/layer5/Conv2D:output:0+Logan/layer5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????!! 2
Logan/layer5/BiasAdd?
Logan/layer5/ReluReluLogan/layer5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????!! 2
Logan/layer5/Relu?
"Logan/layer6/Conv2D/ReadVariableOpReadVariableOp+logan_layer6_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02$
"Logan/layer6/Conv2D/ReadVariableOp?
Logan/layer6/Conv2DConv2DLogan/layer5/Relu:activations:0*Logan/layer6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Logan/layer6/Conv2D?
#Logan/layer6/BiasAdd/ReadVariableOpReadVariableOp,logan_layer6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#Logan/layer6/BiasAdd/ReadVariableOp?
Logan/layer6/BiasAddBiasAddLogan/layer6/Conv2D:output:0+Logan/layer6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Logan/layer6/BiasAdd?
Logan/layer6/ReluReluLogan/layer6/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Logan/layer6/Relu?
Logan/layer7/MaxPoolMaxPoolLogan/layer6/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
Logan/layer7/MaxPool?
Logan/layer7/MaxPool_1MaxPoolLogan/layer7/MaxPool:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
Logan/layer7/MaxPool_1?
Logan/layer8/IdentityIdentityLogan/layer7/MaxPool_1:output:0*
T0*/
_output_shapes
:????????? 2
Logan/layer8/Identityy
Logan/layer9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Logan/layer9/Const?
Logan/layer9/ReshapeReshapeLogan/layer8/Identity:output:0Logan/layer9/Const:output:0*
T0*(
_output_shapes
:??????????2
Logan/layer9/Reshape?
#Logan/layer10/MatMul/ReadVariableOpReadVariableOp,logan_layer10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#Logan/layer10/MatMul/ReadVariableOp?
Logan/layer10/MatMulMatMulLogan/layer9/Reshape:output:0+Logan/layer10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Logan/layer10/MatMul?
$Logan/layer10/BiasAdd/ReadVariableOpReadVariableOp-logan_layer10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$Logan/layer10/BiasAdd/ReadVariableOp?
Logan/layer10/BiasAddBiasAddLogan/layer10/MatMul:product:0,Logan/layer10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Logan/layer10/BiasAdd?
Logan/layer10/ReluReluLogan/layer10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Logan/layer10/Relu?
#Logan/layer11/MatMul/ReadVariableOpReadVariableOp,logan_layer11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#Logan/layer11/MatMul/ReadVariableOp?
Logan/layer11/MatMulMatMul Logan/layer10/Relu:activations:0+Logan/layer11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Logan/layer11/MatMul?
$Logan/layer11/BiasAdd/ReadVariableOpReadVariableOp-logan_layer11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$Logan/layer11/BiasAdd/ReadVariableOp?
Logan/layer11/BiasAddBiasAddLogan/layer11/MatMul:product:0,Logan/layer11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Logan/layer11/BiasAdd?
Logan/layer11/ReluReluLogan/layer11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Logan/layer11/Relu?
"Logan/symbol/MatMul/ReadVariableOpReadVariableOp+logan_symbol_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"Logan/symbol/MatMul/ReadVariableOp?
Logan/symbol/MatMulMatMul Logan/layer11/Relu:activations:0*Logan/symbol/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Logan/symbol/MatMul?
#Logan/symbol/BiasAdd/ReadVariableOpReadVariableOp,logan_symbol_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Logan/symbol/BiasAdd/ReadVariableOp?
Logan/symbol/BiasAddBiasAddLogan/symbol/MatMul:product:0+Logan/symbol/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Logan/symbol/BiasAdd?
Logan/symbol/SoftmaxSoftmaxLogan/symbol/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Logan/symbol/Softmax?
IdentityIdentityLogan/symbol/Softmax:softmax:0$^Logan/layer1/BiasAdd/ReadVariableOp#^Logan/layer1/Conv2D/ReadVariableOp%^Logan/layer10/BiasAdd/ReadVariableOp$^Logan/layer10/MatMul/ReadVariableOp%^Logan/layer11/BiasAdd/ReadVariableOp$^Logan/layer11/MatMul/ReadVariableOp$^Logan/layer2/BiasAdd/ReadVariableOp#^Logan/layer2/Conv2D/ReadVariableOp$^Logan/layer5/BiasAdd/ReadVariableOp#^Logan/layer5/Conv2D/ReadVariableOp$^Logan/layer6/BiasAdd/ReadVariableOp#^Logan/layer6/Conv2D/ReadVariableOp$^Logan/symbol/BiasAdd/ReadVariableOp#^Logan/symbol/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2J
#Logan/layer1/BiasAdd/ReadVariableOp#Logan/layer1/BiasAdd/ReadVariableOp2H
"Logan/layer1/Conv2D/ReadVariableOp"Logan/layer1/Conv2D/ReadVariableOp2L
$Logan/layer10/BiasAdd/ReadVariableOp$Logan/layer10/BiasAdd/ReadVariableOp2J
#Logan/layer10/MatMul/ReadVariableOp#Logan/layer10/MatMul/ReadVariableOp2L
$Logan/layer11/BiasAdd/ReadVariableOp$Logan/layer11/BiasAdd/ReadVariableOp2J
#Logan/layer11/MatMul/ReadVariableOp#Logan/layer11/MatMul/ReadVariableOp2J
#Logan/layer2/BiasAdd/ReadVariableOp#Logan/layer2/BiasAdd/ReadVariableOp2H
"Logan/layer2/Conv2D/ReadVariableOp"Logan/layer2/Conv2D/ReadVariableOp2J
#Logan/layer5/BiasAdd/ReadVariableOp#Logan/layer5/BiasAdd/ReadVariableOp2H
"Logan/layer5/Conv2D/ReadVariableOp"Logan/layer5/Conv2D/ReadVariableOp2J
#Logan/layer6/BiasAdd/ReadVariableOp#Logan/layer6/BiasAdd/ReadVariableOp2H
"Logan/layer6/Conv2D/ReadVariableOp"Logan/layer6/Conv2D/ReadVariableOp2J
#Logan/symbol/BiasAdd/ReadVariableOp#Logan/symbol/BiasAdd/ReadVariableOp2H
"Logan/symbol/MatMul/ReadVariableOp"Logan/symbol/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?

?
B__inference_layer2_layer_call_and_return_conditional_losses_116395

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
C
'__inference_layer8_layer_call_fn_117209

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_1165162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
C__inference_layer11_layer_call_and_return_conditional_losses_117251

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
|
'__inference_layer6_layer_call_fn_117182

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_1164812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????!! ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????!! 
 
_user_specified_nameinputs
?	
?
C__inference_layer10_layer_call_and_return_conditional_losses_117231

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_layer8_layer_call_and_return_conditional_losses_116516

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
B__inference_layer4_layer_call_and_return_conditional_losses_117127

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????$$2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????$$*
dtype0*

seed{2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????$$2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????$$2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????$$2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????$$2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$$:W S
/
_output_shapes
:?????????$$
 
_user_specified_nameinputs
?

?
B__inference_layer1_layer_call_and_return_conditional_losses_116368

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
C
'__inference_layer3_layer_call_fn_116341

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_1163352
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_7:
serving_default_input_7:0???????????:
symbol0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?m
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
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?i
_tf_keras_network?h{"class_name": "Functional", "name": "Logan", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Logan", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 150, 150, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "layer3", "inbound_nodes": [[["layer2", 0, 0, {}]], [["layer3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "layer4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "layer4", "inbound_nodes": [[["layer3", 1, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer5", "inbound_nodes": [[["layer4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer6", "inbound_nodes": [[["layer5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "layer7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "layer7", "inbound_nodes": [[["layer6", 0, 0, {}]], [["layer7", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "layer8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "layer8", "inbound_nodes": [[["layer7", 1, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "layer9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "layer9", "inbound_nodes": [[["layer8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer10", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer10", "inbound_nodes": [[["layer9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer11", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer11", "inbound_nodes": [[["layer10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "symbol", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "symbol", "inbound_nodes": [[["layer11", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["symbol", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 150, 150, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Logan", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 150, 150, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "layer3", "inbound_nodes": [[["layer2", 0, 0, {}]], [["layer3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "layer4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "layer4", "inbound_nodes": [[["layer3", 1, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer5", "inbound_nodes": [[["layer4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer6", "inbound_nodes": [[["layer5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "layer7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "layer7", "inbound_nodes": [[["layer6", 0, 0, {}]], [["layer7", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "layer8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "layer8", "inbound_nodes": [[["layer7", 1, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "layer9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "layer9", "inbound_nodes": [[["layer8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer10", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer10", "inbound_nodes": [[["layer9", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer11", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer11", "inbound_nodes": [[["layer10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "symbol", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "symbol", "inbound_nodes": [[["layer11", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["symbol", 0, 0]]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "top_k_categorical_accuracy", "dtype": "float32", "fn": "top_k_categorical_accuracy"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": [0.01], "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 150, 150, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 150, 150, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150, 150, 1]}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 147, 147, 16]}}
?
 regularization_losses
!trainable_variables
"	variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "layer4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer4", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?	

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 36, 16]}}
?	

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
?
4regularization_losses
5trainable_variables
6	variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "layer7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
8regularization_losses
9trainable_variables
:	variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "layer8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer8", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
<regularization_losses
=trainable_variables
>	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "layer9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

@kernel
Abias
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer10", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1568}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1568]}}
?

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer11", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "symbol", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "symbol", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem?m?m?m?(m?)m?.m?/m?@m?Am?Fm?Gm?Lm?Mm?v?v?v?v?(v?)v?.v?/v?@v?Av?Fv?Gv?Lv?Mv?"
	optimizer
 "
trackable_list_wrapper
?
0
1
2
3
(4
)5
.6
/7
@8
A9
F10
G11
L12
M13"
trackable_list_wrapper
?
0
1
2
3
(4
)5
.6
/7
@8
A9
F10
G11
L12
M13"
trackable_list_wrapper
?
Wlayer_regularization_losses
regularization_losses

Xlayers
trainable_variables
	variables
Ynon_trainable_variables
Zlayer_metrics
[metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':%2layer1/kernel
:2layer1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
\layer_regularization_losses
regularization_losses

]layers
trainable_variables
	variables
^non_trainable_variables
_layer_metrics
`metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2layer2/kernel
:2layer2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
alayer_regularization_losses
regularization_losses

blayers
trainable_variables
	variables
cnon_trainable_variables
dlayer_metrics
emetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
flayer_regularization_losses
 regularization_losses

glayers
!trainable_variables
"	variables
hnon_trainable_variables
ilayer_metrics
jmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
klayer_regularization_losses
$regularization_losses

llayers
%trainable_variables
&	variables
mnon_trainable_variables
nlayer_metrics
ometrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% 2layer5/kernel
: 2layer5/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
player_regularization_losses
*regularization_losses

qlayers
+trainable_variables
,	variables
rnon_trainable_variables
slayer_metrics
tmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%  2layer6/kernel
: 2layer6/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
ulayer_regularization_losses
0regularization_losses

vlayers
1trainable_variables
2	variables
wnon_trainable_variables
xlayer_metrics
ymetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
zlayer_regularization_losses
4regularization_losses

{layers
5trainable_variables
6	variables
|non_trainable_variables
}layer_metrics
~metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
layer_regularization_losses
8regularization_losses
?layers
9trainable_variables
:	variables
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
<regularization_losses
?layers
=trainable_variables
>	variables
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2layer10/kernel
:?2layer10/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Bregularization_losses
?layers
Ctrainable_variables
D	variables
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2layer11/kernel
:?2layer11/bias
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Hregularization_losses
?layers
Itrainable_variables
J	variables
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2symbol/kernel
:2symbol/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Nregularization_losses
?layers
Otrainable_variables
P	variables
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
~
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
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
?0
?1
?2"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "top_k_categorical_accuracy", "dtype": "float32", "config": {"name": "top_k_categorical_accuracy", "dtype": "float32", "fn": "top_k_categorical_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:*2Adam/layer1/kernel/m
:2Adam/layer1/bias/m
,:*2Adam/layer2/kernel/m
:2Adam/layer2/bias/m
,:* 2Adam/layer5/kernel/m
: 2Adam/layer5/bias/m
,:*  2Adam/layer6/kernel/m
: 2Adam/layer6/bias/m
':%
??2Adam/layer10/kernel/m
 :?2Adam/layer10/bias/m
':%
??2Adam/layer11/kernel/m
 :?2Adam/layer11/bias/m
%:#	?2Adam/symbol/kernel/m
:2Adam/symbol/bias/m
,:*2Adam/layer1/kernel/v
:2Adam/layer1/bias/v
,:*2Adam/layer2/kernel/v
:2Adam/layer2/bias/v
,:* 2Adam/layer5/kernel/v
: 2Adam/layer5/bias/v
,:*  2Adam/layer6/kernel/v
: 2Adam/layer6/bias/v
':%
??2Adam/layer10/kernel/v
 :?2Adam/layer10/bias/v
':%
??2Adam/layer11/kernel/v
 :?2Adam/layer11/bias/v
%:#	?2Adam/symbol/kernel/v
:2Adam/symbol/bias/v
?2?
&__inference_Logan_layer_call_fn_116830
&__inference_Logan_layer_call_fn_116751
&__inference_Logan_layer_call_fn_117075
&__inference_Logan_layer_call_fn_117042?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_116329?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_7???????????
?2?
A__inference_Logan_layer_call_and_return_conditional_losses_116948
A__inference_Logan_layer_call_and_return_conditional_losses_116625
A__inference_Logan_layer_call_and_return_conditional_losses_117009
A__inference_Logan_layer_call_and_return_conditional_losses_116671?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_layer1_layer_call_fn_117095?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer1_layer_call_and_return_conditional_losses_117086?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer2_layer_call_fn_117115?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer2_layer_call_and_return_conditional_losses_117106?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer3_layer_call_fn_116341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_layer3_layer_call_and_return_conditional_losses_116335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
'__inference_layer4_layer_call_fn_117142
'__inference_layer4_layer_call_fn_117137?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_layer4_layer_call_and_return_conditional_losses_117127
B__inference_layer4_layer_call_and_return_conditional_losses_117132?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_layer5_layer_call_fn_117162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer5_layer_call_and_return_conditional_losses_117153?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer6_layer_call_fn_117182?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer6_layer_call_and_return_conditional_losses_117173?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_layer7_layer_call_fn_116353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_layer7_layer_call_and_return_conditional_losses_116347?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
'__inference_layer8_layer_call_fn_117209
'__inference_layer8_layer_call_fn_117204?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_layer8_layer_call_and_return_conditional_losses_117194
B__inference_layer8_layer_call_and_return_conditional_losses_117199?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_layer9_layer_call_fn_117220?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_layer9_layer_call_and_return_conditional_losses_117215?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_layer10_layer_call_fn_117240?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_layer10_layer_call_and_return_conditional_losses_117231?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_layer11_layer_call_fn_117260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_layer11_layer_call_and_return_conditional_losses_117251?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_symbol_layer_call_fn_117280?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_symbol_layer_call_and_return_conditional_losses_117271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_116873input_7"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
A__inference_Logan_layer_call_and_return_conditional_losses_116625{()./@AFGLMB??
8?5
+?(
input_7???????????
p

 
? "%?"
?
0?????????
? ?
A__inference_Logan_layer_call_and_return_conditional_losses_116671{()./@AFGLMB??
8?5
+?(
input_7???????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_Logan_layer_call_and_return_conditional_losses_116948z()./@AFGLMA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
A__inference_Logan_layer_call_and_return_conditional_losses_117009z()./@AFGLMA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
&__inference_Logan_layer_call_fn_116751n()./@AFGLMB??
8?5
+?(
input_7???????????
p

 
? "???????????
&__inference_Logan_layer_call_fn_116830n()./@AFGLMB??
8?5
+?(
input_7???????????
p 

 
? "???????????
&__inference_Logan_layer_call_fn_117042m()./@AFGLMA?>
7?4
*?'
inputs???????????
p

 
? "???????????
&__inference_Logan_layer_call_fn_117075m()./@AFGLMA?>
7?4
*?'
inputs???????????
p 

 
? "???????????
!__inference__wrapped_model_116329}()./@AFGLM:?7
0?-
+?(
input_7???????????
? "/?,
*
symbol ?
symbol??????????
C__inference_layer10_layer_call_and_return_conditional_losses_117231^@A0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_layer10_layer_call_fn_117240Q@A0?-
&?#
!?
inputs??????????
? "????????????
C__inference_layer11_layer_call_and_return_conditional_losses_117251^FG0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_layer11_layer_call_fn_117260QFG0?-
&?#
!?
inputs??????????
? "????????????
B__inference_layer1_layer_call_and_return_conditional_losses_117086p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
'__inference_layer1_layer_call_fn_117095c9?6
/?,
*?'
inputs???????????
? ""?????????????
B__inference_layer2_layer_call_and_return_conditional_losses_117106p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
'__inference_layer2_layer_call_fn_117115c9?6
/?,
*?'
inputs???????????
? ""?????????????
B__inference_layer3_layer_call_and_return_conditional_losses_116335?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_layer3_layer_call_fn_116341?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_layer4_layer_call_and_return_conditional_losses_117127l;?8
1?.
(?%
inputs?????????$$
p
? "-?*
#? 
0?????????$$
? ?
B__inference_layer4_layer_call_and_return_conditional_losses_117132l;?8
1?.
(?%
inputs?????????$$
p 
? "-?*
#? 
0?????????$$
? ?
'__inference_layer4_layer_call_fn_117137_;?8
1?.
(?%
inputs?????????$$
p
? " ??????????$$?
'__inference_layer4_layer_call_fn_117142_;?8
1?.
(?%
inputs?????????$$
p 
? " ??????????$$?
B__inference_layer5_layer_call_and_return_conditional_losses_117153l()7?4
-?*
(?%
inputs?????????$$
? "-?*
#? 
0?????????!! 
? ?
'__inference_layer5_layer_call_fn_117162_()7?4
-?*
(?%
inputs?????????$$
? " ??????????!! ?
B__inference_layer6_layer_call_and_return_conditional_losses_117173l./7?4
-?*
(?%
inputs?????????!! 
? "-?*
#? 
0????????? 
? ?
'__inference_layer6_layer_call_fn_117182_./7?4
-?*
(?%
inputs?????????!! 
? " ?????????? ?
B__inference_layer7_layer_call_and_return_conditional_losses_116347?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_layer7_layer_call_fn_116353?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_layer8_layer_call_and_return_conditional_losses_117194l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
B__inference_layer8_layer_call_and_return_conditional_losses_117199l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
'__inference_layer8_layer_call_fn_117204_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
'__inference_layer8_layer_call_fn_117209_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
B__inference_layer9_layer_call_and_return_conditional_losses_117215a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? 
'__inference_layer9_layer_call_fn_117220T7?4
-?*
(?%
inputs????????? 
? "????????????
$__inference_signature_wrapper_116873?()./@AFGLME?B
? 
;?8
6
input_7+?(
input_7???????????"/?,
*
symbol ?
symbol??????????
B__inference_symbol_layer_call_and_return_conditional_losses_117271]LM0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_symbol_layer_call_fn_117280PLM0?-
&?#
!?
inputs??????????
? "??????????