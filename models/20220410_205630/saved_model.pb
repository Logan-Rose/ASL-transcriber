??	
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
executor_typestring ??
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
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68̹
~
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namelayer1/kernel
w
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*&
_output_shapes
:@*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:@*
dtype0
~
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *
shared_namelayer2/kernel
w
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*&
_output_shapes
:@ *
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
: *
dtype0
~
layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_namelayer4/kernel
w
!layer4/kernel/Read/ReadVariableOpReadVariableOplayer4/kernel*&
_output_shapes
: @*
dtype0
n
layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namelayer4/bias
g
layer4/bias/Read/ReadVariableOpReadVariableOplayer4/bias*
_output_shapes
:@*
dtype0
~
layer5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_namelayer5/kernel
w
!layer5/kernel/Read/ReadVariableOpReadVariableOplayer5/kernel*&
_output_shapes
:@@*
dtype0
n
layer5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namelayer5/bias
g
layer5/bias/Read/ReadVariableOpReadVariableOplayer5/bias*
_output_shapes
:@*
dtype0
y
layer7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_namelayer7/kernel
r
!layer7/kernel/Read/ReadVariableOpReadVariableOplayer7/kernel*!
_output_shapes
:???*
dtype0
o
layer7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer7/bias
h
layer7/bias/Read/ReadVariableOpReadVariableOplayer7/bias*
_output_shapes	
:?*
dtype0
x
layer8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namelayer8/kernel
q
!layer8/kernel/Read/ReadVariableOpReadVariableOplayer8/kernel* 
_output_shapes
:
??*
dtype0
o
layer8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer8/bias
h
layer8/bias/Read/ReadVariableOpReadVariableOplayer8/bias*
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
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
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
RMSprop/layer1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameRMSprop/layer1/kernel/rms
?
-RMSprop/layer1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer1/kernel/rms*&
_output_shapes
:@*
dtype0
?
RMSprop/layer1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameRMSprop/layer1/bias/rms

+RMSprop/layer1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer1/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/layer2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ **
shared_nameRMSprop/layer2/kernel/rms
?
-RMSprop/layer2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer2/kernel/rms*&
_output_shapes
:@ *
dtype0
?
RMSprop/layer2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameRMSprop/layer2/bias/rms

+RMSprop/layer2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer2/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/layer4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameRMSprop/layer4/kernel/rms
?
-RMSprop/layer4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer4/kernel/rms*&
_output_shapes
: @*
dtype0
?
RMSprop/layer4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameRMSprop/layer4/bias/rms

+RMSprop/layer4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer4/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/layer5/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_nameRMSprop/layer5/kernel/rms
?
-RMSprop/layer5/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer5/kernel/rms*&
_output_shapes
:@@*
dtype0
?
RMSprop/layer5/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameRMSprop/layer5/bias/rms

+RMSprop/layer5/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer5/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/layer7/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:???**
shared_nameRMSprop/layer7/kernel/rms
?
-RMSprop/layer7/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer7/kernel/rms*!
_output_shapes
:???*
dtype0
?
RMSprop/layer7/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameRMSprop/layer7/bias/rms
?
+RMSprop/layer7/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer7/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/layer8/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameRMSprop/layer8/kernel/rms
?
-RMSprop/layer8/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer8/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/layer8/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameRMSprop/layer8/bias/rms
?
+RMSprop/layer8/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer8/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/symbol/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameRMSprop/symbol/kernel/rms
?
-RMSprop/symbol/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/symbol/kernel/rms*
_output_shapes
:	?*
dtype0
?
RMSprop/symbol/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameRMSprop/symbol/bias/rms

+RMSprop/symbol/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/symbol/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
?S
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?R
value?RB?R B?R
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
?

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
?

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
?

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses*
?

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
?
Xiter
	Ydecay
Zlearning_rate
[momentum
\rho
rms?
rms?
rms?
rms?
*rms?
+rms?
2rms?
3rms?
@rms?
Arms?
Hrms?
Irms?
Prms?
Qrms?*
j
0
1
2
3
*4
+5
26
37
@8
A9
H10
I11
P12
Q13*
j
0
1
2
3
*4
+5
26
37
@8
A9
H10
I11
P12
Q13*
* 
?
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

bserving_default* 
]W
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUElayer4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUElayer5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

20
31*

20
31*
* 
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUElayer7/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer7/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUElayer8/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer8/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

H0
I1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEsymbol/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEsymbol/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

P0
Q1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
J
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
9*

?0
?1
?2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
??
VARIABLE_VALUERMSprop/layer1/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/layer1/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/layer2/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/layer2/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/layer4/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/layer4/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/layer5/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/layer5/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/layer7/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/layer7/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/layer8/kernel/rmsTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/layer8/bias/rmsRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/symbol/kernel/rmsTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/symbol/bias/rmsRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????dd*
dtype0*$
shape:?????????dd
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer1/kernellayer1/biaslayer2/kernellayer2/biaslayer4/kernellayer4/biaslayer5/kernellayer5/biaslayer7/kernellayer7/biaslayer8/kernellayer8/biassymbol/kernelsymbol/bias*
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
GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_33749
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer4/kernel/Read/ReadVariableOplayer4/bias/Read/ReadVariableOp!layer5/kernel/Read/ReadVariableOplayer5/bias/Read/ReadVariableOp!layer7/kernel/Read/ReadVariableOplayer7/bias/Read/ReadVariableOp!layer8/kernel/Read/ReadVariableOplayer8/bias/Read/ReadVariableOp!symbol/kernel/Read/ReadVariableOpsymbol/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp-RMSprop/layer1/kernel/rms/Read/ReadVariableOp+RMSprop/layer1/bias/rms/Read/ReadVariableOp-RMSprop/layer2/kernel/rms/Read/ReadVariableOp+RMSprop/layer2/bias/rms/Read/ReadVariableOp-RMSprop/layer4/kernel/rms/Read/ReadVariableOp+RMSprop/layer4/bias/rms/Read/ReadVariableOp-RMSprop/layer5/kernel/rms/Read/ReadVariableOp+RMSprop/layer5/bias/rms/Read/ReadVariableOp-RMSprop/layer7/kernel/rms/Read/ReadVariableOp+RMSprop/layer7/bias/rms/Read/ReadVariableOp-RMSprop/layer8/kernel/rms/Read/ReadVariableOp+RMSprop/layer8/bias/rms/Read/ReadVariableOp-RMSprop/symbol/kernel/rms/Read/ReadVariableOp+RMSprop/symbol/bias/rms/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_34050
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer4/kernellayer4/biaslayer5/kernellayer5/biaslayer7/kernellayer7/biaslayer8/kernellayer8/biassymbol/kernelsymbol/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1total_2count_2RMSprop/layer1/kernel/rmsRMSprop/layer1/bias/rmsRMSprop/layer2/kernel/rmsRMSprop/layer2/bias/rmsRMSprop/layer4/kernel/rmsRMSprop/layer4/bias/rmsRMSprop/layer5/kernel/rmsRMSprop/layer5/bias/rmsRMSprop/layer7/kernel/rmsRMSprop/layer7/bias/rmsRMSprop/layer8/kernel/rmsRMSprop/layer8/bias/rmsRMSprop/symbol/kernel/rmsRMSprop/symbol/bias/rms*3
Tin,
*2(*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_34177??
?
?
A__inference_layer4_layer_call_and_return_conditional_losses_33118

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????**@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????**@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????-- : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????-- 
 
_user_specified_nameinputs
?
?
A__inference_layer1_layer_call_and_return_conditional_losses_33769

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????__@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????__@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?)
?
@__inference_Logan_layer_call_and_return_conditional_losses_33530
input_1&
layer1_33492:@
layer1_33494:@&
layer2_33497:@ 
layer2_33499: &
layer4_33503: @
layer4_33505:@&
layer5_33508:@@
layer5_33510:@!
layer7_33514:???
layer7_33516:	? 
layer8_33519:
??
layer8_33521:	?
symbol_33524:	?
symbol_33526:
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer4/StatefulPartitionedCall?layer5/StatefulPartitionedCall?layer7/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_1layer1_33492layer1_33494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????__@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_33083?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_33497layer2_33499*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_33100?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_33062?
layer4/StatefulPartitionedCallStatefulPartitionedCalllayer3/PartitionedCall:output:0layer4_33503layer4_33505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_33118?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_33508layer5_33510*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????''@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_33135?
layer6/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer6_layer_call_and_return_conditional_losses_33147?
layer7/StatefulPartitionedCallStatefulPartitionedCalllayer6/PartitionedCall:output:0layer7_33514layer7_33516*
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
GPU 2J 8? *J
fERC
A__inference_layer7_layer_call_and_return_conditional_losses_33160?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_33519layer8_33521*
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
GPU 2J 8? *J
fERC
A__inference_layer8_layer_call_and_return_conditional_losses_33177?
symbol/StatefulPartitionedCallStatefulPartitionedCall'layer8/StatefulPartitionedCall:output:0symbol_33524symbol_33526*
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
GPU 2J 8? *J
fERC
A__inference_symbol_layer_call_and_return_conditional_losses_33194v
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer7/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_1
?

?
A__inference_layer8_layer_call_and_return_conditional_losses_33177

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_33749
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *)
f$R"
 __inference__wrapped_model_33053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_1
?
?
A__inference_layer2_layer_call_and_return_conditional_losses_33100

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????ZZ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????ZZ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????__@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????__@
 
_user_specified_nameinputs
?)
?
@__inference_Logan_layer_call_and_return_conditional_losses_33201

inputs&
layer1_33084:@
layer1_33086:@&
layer2_33101:@ 
layer2_33103: &
layer4_33119: @
layer4_33121:@&
layer5_33136:@@
layer5_33138:@!
layer7_33161:???
layer7_33163:	? 
layer8_33178:
??
layer8_33180:	?
symbol_33195:	?
symbol_33197:
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer4/StatefulPartitionedCall?layer5/StatefulPartitionedCall?layer7/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_33084layer1_33086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????__@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_33083?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_33101layer2_33103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_33100?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_33062?
layer4/StatefulPartitionedCallStatefulPartitionedCalllayer3/PartitionedCall:output:0layer4_33119layer4_33121*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_33118?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_33136layer5_33138*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????''@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_33135?
layer6/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer6_layer_call_and_return_conditional_losses_33147?
layer7/StatefulPartitionedCallStatefulPartitionedCalllayer6/PartitionedCall:output:0layer7_33161layer7_33163*
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
GPU 2J 8? *J
fERC
A__inference_layer7_layer_call_and_return_conditional_losses_33160?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_33178layer8_33180*
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
GPU 2J 8? *J
fERC
A__inference_layer8_layer_call_and_return_conditional_losses_33177?
symbol/StatefulPartitionedCallStatefulPartitionedCall'layer8/StatefulPartitionedCall:output:0symbol_33195symbol_33197*
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
GPU 2J 8? *J
fERC
A__inference_symbol_layer_call_and_return_conditional_losses_33194v
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer7/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?@
?

@__inference_Logan_layer_call_and_return_conditional_losses_33714

inputs?
%layer1_conv2d_readvariableop_resource:@4
&layer1_biasadd_readvariableop_resource:@?
%layer2_conv2d_readvariableop_resource:@ 4
&layer2_biasadd_readvariableop_resource: ?
%layer4_conv2d_readvariableop_resource: @4
&layer4_biasadd_readvariableop_resource:@?
%layer5_conv2d_readvariableop_resource:@@4
&layer5_biasadd_readvariableop_resource:@:
%layer7_matmul_readvariableop_resource:???5
&layer7_biasadd_readvariableop_resource:	?9
%layer8_matmul_readvariableop_resource:
??5
&layer8_biasadd_readvariableop_resource:	?8
%symbol_matmul_readvariableop_resource:	?4
&symbol_biasadd_readvariableop_resource:
identity??layer1/BiasAdd/ReadVariableOp?layer1/Conv2D/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/Conv2D/ReadVariableOp?layer4/BiasAdd/ReadVariableOp?layer4/Conv2D/ReadVariableOp?layer5/BiasAdd/ReadVariableOp?layer5/Conv2D/ReadVariableOp?layer7/BiasAdd/ReadVariableOp?layer7/MatMul/ReadVariableOp?layer8/BiasAdd/ReadVariableOp?layer8/MatMul/ReadVariableOp?symbol/BiasAdd/ReadVariableOp?symbol/MatMul/ReadVariableOp?
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@*
paddingVALID*
strides
?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@f
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????__@?
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
layer2/Conv2DConv2Dlayer1/Relu:activations:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ *
paddingVALID*
strides
?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ f
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ZZ ?
layer3/MaxPoolMaxPoollayer2/Relu:activations:0*/
_output_shapes
:?????????-- *
ksize
*
paddingVALID*
strides
?
layer4/Conv2D/ReadVariableOpReadVariableOp%layer4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
layer4/Conv2DConv2Dlayer3/MaxPool:output:0$layer4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@*
paddingVALID*
strides
?
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer4/BiasAddBiasAddlayer4/Conv2D:output:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@f
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**@?
layer5/Conv2D/ReadVariableOpReadVariableOp%layer5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
layer5/Conv2DConv2Dlayer4/Relu:activations:0$layer5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@*
paddingVALID*
strides
?
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer5/BiasAddBiasAddlayer5/Conv2D:output:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@f
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????''@]
layer6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@| 
layer6/ReshapeReshapelayer5/Relu:activations:0layer6/Const:output:0*
T0*)
_output_shapes
:????????????
layer7/MatMul/ReadVariableOpReadVariableOp%layer7_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
layer7/MatMulMatMullayer6/Reshape:output:0$layer7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
layer7/BiasAdd/ReadVariableOpReadVariableOp&layer7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer7/BiasAddBiasAddlayer7/MatMul:product:0%layer7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????_
layer7/ReluRelulayer7/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
layer8/MatMul/ReadVariableOpReadVariableOp%layer8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
layer8/MatMulMatMullayer7/Relu:activations:0$layer8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
layer8/BiasAdd/ReadVariableOpReadVariableOp&layer8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer8/BiasAddBiasAddlayer8/MatMul:product:0%layer8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????_
layer8/ReluRelulayer8/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
symbol/MatMul/ReadVariableOpReadVariableOp%symbol_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
symbol/MatMulMatMullayer8/Relu:activations:0$symbol/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
symbol/BiasAdd/ReadVariableOpReadVariableOp&symbol_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
symbol/BiasAddBiasAddsymbol/MatMul:product:0%symbol/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
symbol/SoftmaxSoftmaxsymbol/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitysymbol/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/Conv2D/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/Conv2D/ReadVariableOp^layer7/BiasAdd/ReadVariableOp^layer7/MatMul/ReadVariableOp^layer8/BiasAdd/ReadVariableOp^layer8/MatMul/ReadVariableOp^symbol/BiasAdd/ReadVariableOp^symbol/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/Conv2D/ReadVariableOplayer4/Conv2D/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/Conv2D/ReadVariableOplayer5/Conv2D/ReadVariableOp2>
layer7/BiasAdd/ReadVariableOplayer7/BiasAdd/ReadVariableOp2<
layer7/MatMul/ReadVariableOplayer7/MatMul/ReadVariableOp2>
layer8/BiasAdd/ReadVariableOplayer8/BiasAdd/ReadVariableOp2<
layer8/MatMul/ReadVariableOplayer8/MatMul/ReadVariableOp2>
symbol/BiasAdd/ReadVariableOpsymbol/BiasAdd/ReadVariableOp2<
symbol/MatMul/ReadVariableOpsymbol/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
A__inference_layer2_layer_call_and_return_conditional_losses_33789

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????ZZ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????ZZ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????__@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????__@
 
_user_specified_nameinputs
?
B
&__inference_layer3_layer_call_fn_33794

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
GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_33062?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_layer5_layer_call_and_return_conditional_losses_33839

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????''@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????''@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????**@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**@
 
_user_specified_nameinputs
?
?
%__inference_Logan_layer_call_fn_33232
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *I
fDRB
@__inference_Logan_layer_call_and_return_conditional_losses_33201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_1
?
]
A__inference_layer3_layer_call_and_return_conditional_losses_33062

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_layer4_layer_call_and_return_conditional_losses_33819

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????**@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????**@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????-- : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????-- 
 
_user_specified_nameinputs
?
?
&__inference_layer4_layer_call_fn_33808

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_33118w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????**@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????-- : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-- 
 
_user_specified_nameinputs
?)
?
@__inference_Logan_layer_call_and_return_conditional_losses_33384

inputs&
layer1_33346:@
layer1_33348:@&
layer2_33351:@ 
layer2_33353: &
layer4_33357: @
layer4_33359:@&
layer5_33362:@@
layer5_33364:@!
layer7_33368:???
layer7_33370:	? 
layer8_33373:
??
layer8_33375:	?
symbol_33378:	?
symbol_33380:
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer4/StatefulPartitionedCall?layer5/StatefulPartitionedCall?layer7/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_33346layer1_33348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????__@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_33083?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_33351layer2_33353*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_33100?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_33062?
layer4/StatefulPartitionedCallStatefulPartitionedCalllayer3/PartitionedCall:output:0layer4_33357layer4_33359*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_33118?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_33362layer5_33364*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????''@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_33135?
layer6/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer6_layer_call_and_return_conditional_losses_33147?
layer7/StatefulPartitionedCallStatefulPartitionedCalllayer6/PartitionedCall:output:0layer7_33368layer7_33370*
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
GPU 2J 8? *J
fERC
A__inference_layer7_layer_call_and_return_conditional_losses_33160?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_33373layer8_33375*
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
GPU 2J 8? *J
fERC
A__inference_layer8_layer_call_and_return_conditional_losses_33177?
symbol/StatefulPartitionedCallStatefulPartitionedCall'layer8/StatefulPartitionedCall:output:0symbol_33378symbol_33380*
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
GPU 2J 8? *J
fERC
A__inference_symbol_layer_call_and_return_conditional_losses_33194v
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer7/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
&__inference_layer2_layer_call_fn_33778

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_33100w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????ZZ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????__@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????__@
 
_user_specified_nameinputs
?
?
%__inference_Logan_layer_call_fn_33602

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
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
GPU 2J 8? *I
fDRB
@__inference_Logan_layer_call_and_return_conditional_losses_33384o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?

?
A__inference_layer8_layer_call_and_return_conditional_losses_33890

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_symbol_layer_call_and_return_conditional_losses_33194

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_layer5_layer_call_fn_33828

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????''@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_33135w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????''@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????**@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????**@
 
_user_specified_nameinputs
?
?
&__inference_layer7_layer_call_fn_33859

inputs
unknown:???
	unknown_0:	?
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
GPU 2J 8? *J
fERC
A__inference_layer7_layer_call_and_return_conditional_losses_33160p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
%__inference_Logan_layer_call_fn_33569

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
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
GPU 2J 8? *I
fDRB
@__inference_Logan_layer_call_and_return_conditional_losses_33201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?)
?
@__inference_Logan_layer_call_and_return_conditional_losses_33489
input_1&
layer1_33451:@
layer1_33453:@&
layer2_33456:@ 
layer2_33458: &
layer4_33462: @
layer4_33464:@&
layer5_33467:@@
layer5_33469:@!
layer7_33473:???
layer7_33475:	? 
layer8_33478:
??
layer8_33480:	?
symbol_33483:	?
symbol_33485:
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer4/StatefulPartitionedCall?layer5/StatefulPartitionedCall?layer7/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_1layer1_33451layer1_33453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????__@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_33083?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_33456layer2_33458*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_33100?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_33062?
layer4/StatefulPartitionedCallStatefulPartitionedCalllayer3/PartitionedCall:output:0layer4_33462layer4_33464*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_33118?
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_33467layer5_33469*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????''@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_33135?
layer6/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer6_layer_call_and_return_conditional_losses_33147?
layer7/StatefulPartitionedCallStatefulPartitionedCalllayer6/PartitionedCall:output:0layer7_33473layer7_33475*
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
GPU 2J 8? *J
fERC
A__inference_layer7_layer_call_and_return_conditional_losses_33160?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_33478layer8_33480*
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
GPU 2J 8? *J
fERC
A__inference_layer8_layer_call_and_return_conditional_losses_33177?
symbol/StatefulPartitionedCallStatefulPartitionedCall'layer8/StatefulPartitionedCall:output:0symbol_33483symbol_33485*
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
GPU 2J 8? *J
fERC
A__inference_symbol_layer_call_and_return_conditional_losses_33194v
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer7/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_1
?
?
%__inference_Logan_layer_call_fn_33448
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:
??

unknown_10:	?

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *I
fDRB
@__inference_Logan_layer_call_and_return_conditional_losses_33384o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_1
?G
?
 __inference__wrapped_model_33053
input_1E
+logan_layer1_conv2d_readvariableop_resource:@:
,logan_layer1_biasadd_readvariableop_resource:@E
+logan_layer2_conv2d_readvariableop_resource:@ :
,logan_layer2_biasadd_readvariableop_resource: E
+logan_layer4_conv2d_readvariableop_resource: @:
,logan_layer4_biasadd_readvariableop_resource:@E
+logan_layer5_conv2d_readvariableop_resource:@@:
,logan_layer5_biasadd_readvariableop_resource:@@
+logan_layer7_matmul_readvariableop_resource:???;
,logan_layer7_biasadd_readvariableop_resource:	??
+logan_layer8_matmul_readvariableop_resource:
??;
,logan_layer8_biasadd_readvariableop_resource:	?>
+logan_symbol_matmul_readvariableop_resource:	?:
,logan_symbol_biasadd_readvariableop_resource:
identity??#Logan/layer1/BiasAdd/ReadVariableOp?"Logan/layer1/Conv2D/ReadVariableOp?#Logan/layer2/BiasAdd/ReadVariableOp?"Logan/layer2/Conv2D/ReadVariableOp?#Logan/layer4/BiasAdd/ReadVariableOp?"Logan/layer4/Conv2D/ReadVariableOp?#Logan/layer5/BiasAdd/ReadVariableOp?"Logan/layer5/Conv2D/ReadVariableOp?#Logan/layer7/BiasAdd/ReadVariableOp?"Logan/layer7/MatMul/ReadVariableOp?#Logan/layer8/BiasAdd/ReadVariableOp?"Logan/layer8/MatMul/ReadVariableOp?#Logan/symbol/BiasAdd/ReadVariableOp?"Logan/symbol/MatMul/ReadVariableOp?
"Logan/layer1/Conv2D/ReadVariableOpReadVariableOp+logan_layer1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Logan/layer1/Conv2DConv2Dinput_1*Logan/layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@*
paddingVALID*
strides
?
#Logan/layer1/BiasAdd/ReadVariableOpReadVariableOp,logan_layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Logan/layer1/BiasAddBiasAddLogan/layer1/Conv2D:output:0+Logan/layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@r
Logan/layer1/ReluReluLogan/layer1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????__@?
"Logan/layer2/Conv2D/ReadVariableOpReadVariableOp+logan_layer2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Logan/layer2/Conv2DConv2DLogan/layer1/Relu:activations:0*Logan/layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ *
paddingVALID*
strides
?
#Logan/layer2/BiasAdd/ReadVariableOpReadVariableOp,logan_layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Logan/layer2/BiasAddBiasAddLogan/layer2/Conv2D:output:0+Logan/layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ r
Logan/layer2/ReluReluLogan/layer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ZZ ?
Logan/layer3/MaxPoolMaxPoolLogan/layer2/Relu:activations:0*/
_output_shapes
:?????????-- *
ksize
*
paddingVALID*
strides
?
"Logan/layer4/Conv2D/ReadVariableOpReadVariableOp+logan_layer4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Logan/layer4/Conv2DConv2DLogan/layer3/MaxPool:output:0*Logan/layer4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@*
paddingVALID*
strides
?
#Logan/layer4/BiasAdd/ReadVariableOpReadVariableOp,logan_layer4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Logan/layer4/BiasAddBiasAddLogan/layer4/Conv2D:output:0+Logan/layer4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@r
Logan/layer4/ReluReluLogan/layer4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**@?
"Logan/layer5/Conv2D/ReadVariableOpReadVariableOp+logan_layer5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Logan/layer5/Conv2DConv2DLogan/layer4/Relu:activations:0*Logan/layer5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@*
paddingVALID*
strides
?
#Logan/layer5/BiasAdd/ReadVariableOpReadVariableOp,logan_layer5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Logan/layer5/BiasAddBiasAddLogan/layer5/Conv2D:output:0+Logan/layer5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@r
Logan/layer5/ReluReluLogan/layer5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????''@c
Logan/layer6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@| ?
Logan/layer6/ReshapeReshapeLogan/layer5/Relu:activations:0Logan/layer6/Const:output:0*
T0*)
_output_shapes
:????????????
"Logan/layer7/MatMul/ReadVariableOpReadVariableOp+logan_layer7_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
Logan/layer7/MatMulMatMulLogan/layer6/Reshape:output:0*Logan/layer7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#Logan/layer7/BiasAdd/ReadVariableOpReadVariableOp,logan_layer7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Logan/layer7/BiasAddBiasAddLogan/layer7/MatMul:product:0+Logan/layer7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
Logan/layer7/ReluReluLogan/layer7/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"Logan/layer8/MatMul/ReadVariableOpReadVariableOp+logan_layer8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
Logan/layer8/MatMulMatMulLogan/layer7/Relu:activations:0*Logan/layer8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
#Logan/layer8/BiasAdd/ReadVariableOpReadVariableOp,logan_layer8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Logan/layer8/BiasAddBiasAddLogan/layer8/MatMul:product:0+Logan/layer8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????k
Logan/layer8/ReluReluLogan/layer8/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"Logan/symbol/MatMul/ReadVariableOpReadVariableOp+logan_symbol_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Logan/symbol/MatMulMatMulLogan/layer8/Relu:activations:0*Logan/symbol/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#Logan/symbol/BiasAdd/ReadVariableOpReadVariableOp,logan_symbol_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Logan/symbol/BiasAddBiasAddLogan/symbol/MatMul:product:0+Logan/symbol/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
Logan/symbol/SoftmaxSoftmaxLogan/symbol/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m
IdentityIdentityLogan/symbol/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^Logan/layer1/BiasAdd/ReadVariableOp#^Logan/layer1/Conv2D/ReadVariableOp$^Logan/layer2/BiasAdd/ReadVariableOp#^Logan/layer2/Conv2D/ReadVariableOp$^Logan/layer4/BiasAdd/ReadVariableOp#^Logan/layer4/Conv2D/ReadVariableOp$^Logan/layer5/BiasAdd/ReadVariableOp#^Logan/layer5/Conv2D/ReadVariableOp$^Logan/layer7/BiasAdd/ReadVariableOp#^Logan/layer7/MatMul/ReadVariableOp$^Logan/layer8/BiasAdd/ReadVariableOp#^Logan/layer8/MatMul/ReadVariableOp$^Logan/symbol/BiasAdd/ReadVariableOp#^Logan/symbol/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 2J
#Logan/layer1/BiasAdd/ReadVariableOp#Logan/layer1/BiasAdd/ReadVariableOp2H
"Logan/layer1/Conv2D/ReadVariableOp"Logan/layer1/Conv2D/ReadVariableOp2J
#Logan/layer2/BiasAdd/ReadVariableOp#Logan/layer2/BiasAdd/ReadVariableOp2H
"Logan/layer2/Conv2D/ReadVariableOp"Logan/layer2/Conv2D/ReadVariableOp2J
#Logan/layer4/BiasAdd/ReadVariableOp#Logan/layer4/BiasAdd/ReadVariableOp2H
"Logan/layer4/Conv2D/ReadVariableOp"Logan/layer4/Conv2D/ReadVariableOp2J
#Logan/layer5/BiasAdd/ReadVariableOp#Logan/layer5/BiasAdd/ReadVariableOp2H
"Logan/layer5/Conv2D/ReadVariableOp"Logan/layer5/Conv2D/ReadVariableOp2J
#Logan/layer7/BiasAdd/ReadVariableOp#Logan/layer7/BiasAdd/ReadVariableOp2H
"Logan/layer7/MatMul/ReadVariableOp"Logan/layer7/MatMul/ReadVariableOp2J
#Logan/layer8/BiasAdd/ReadVariableOp#Logan/layer8/BiasAdd/ReadVariableOp2H
"Logan/layer8/MatMul/ReadVariableOp"Logan/layer8/MatMul/ReadVariableOp2J
#Logan/symbol/BiasAdd/ReadVariableOp#Logan/symbol/BiasAdd/ReadVariableOp2H
"Logan/symbol/MatMul/ReadVariableOp"Logan/symbol/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_1
?

?
A__inference_layer7_layer_call_and_return_conditional_losses_33160

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_symbol_layer_call_fn_33899

inputs
unknown:	?
	unknown_0:
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
GPU 2J 8? *J
fERC
A__inference_symbol_layer_call_and_return_conditional_losses_33194o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?@
?

@__inference_Logan_layer_call_and_return_conditional_losses_33658

inputs?
%layer1_conv2d_readvariableop_resource:@4
&layer1_biasadd_readvariableop_resource:@?
%layer2_conv2d_readvariableop_resource:@ 4
&layer2_biasadd_readvariableop_resource: ?
%layer4_conv2d_readvariableop_resource: @4
&layer4_biasadd_readvariableop_resource:@?
%layer5_conv2d_readvariableop_resource:@@4
&layer5_biasadd_readvariableop_resource:@:
%layer7_matmul_readvariableop_resource:???5
&layer7_biasadd_readvariableop_resource:	?9
%layer8_matmul_readvariableop_resource:
??5
&layer8_biasadd_readvariableop_resource:	?8
%symbol_matmul_readvariableop_resource:	?4
&symbol_biasadd_readvariableop_resource:
identity??layer1/BiasAdd/ReadVariableOp?layer1/Conv2D/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/Conv2D/ReadVariableOp?layer4/BiasAdd/ReadVariableOp?layer4/Conv2D/ReadVariableOp?layer5/BiasAdd/ReadVariableOp?layer5/Conv2D/ReadVariableOp?layer7/BiasAdd/ReadVariableOp?layer7/MatMul/ReadVariableOp?layer8/BiasAdd/ReadVariableOp?layer8/MatMul/ReadVariableOp?symbol/BiasAdd/ReadVariableOp?symbol/MatMul/ReadVariableOp?
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@*
paddingVALID*
strides
?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@f
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????__@?
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
layer2/Conv2DConv2Dlayer1/Relu:activations:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ *
paddingVALID*
strides
?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ZZ f
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ZZ ?
layer3/MaxPoolMaxPoollayer2/Relu:activations:0*/
_output_shapes
:?????????-- *
ksize
*
paddingVALID*
strides
?
layer4/Conv2D/ReadVariableOpReadVariableOp%layer4_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
layer4/Conv2DConv2Dlayer3/MaxPool:output:0$layer4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@*
paddingVALID*
strides
?
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer4/BiasAddBiasAddlayer4/Conv2D:output:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????**@f
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????**@?
layer5/Conv2D/ReadVariableOpReadVariableOp%layer5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
layer5/Conv2DConv2Dlayer4/Relu:activations:0$layer5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@*
paddingVALID*
strides
?
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer5/BiasAddBiasAddlayer5/Conv2D:output:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@f
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????''@]
layer6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@| 
layer6/ReshapeReshapelayer5/Relu:activations:0layer6/Const:output:0*
T0*)
_output_shapes
:????????????
layer7/MatMul/ReadVariableOpReadVariableOp%layer7_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
layer7/MatMulMatMullayer6/Reshape:output:0$layer7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
layer7/BiasAdd/ReadVariableOpReadVariableOp&layer7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer7/BiasAddBiasAddlayer7/MatMul:product:0%layer7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????_
layer7/ReluRelulayer7/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
layer8/MatMul/ReadVariableOpReadVariableOp%layer8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
layer8/MatMulMatMullayer7/Relu:activations:0$layer8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
layer8/BiasAdd/ReadVariableOpReadVariableOp&layer8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
layer8/BiasAddBiasAddlayer8/MatMul:product:0%layer8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????_
layer8/ReluRelulayer8/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
symbol/MatMul/ReadVariableOpReadVariableOp%symbol_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
symbol/MatMulMatMullayer8/Relu:activations:0$symbol/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
symbol/BiasAdd/ReadVariableOpReadVariableOp&symbol_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
symbol/BiasAddBiasAddsymbol/MatMul:product:0%symbol/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
symbol/SoftmaxSoftmaxsymbol/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitysymbol/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/Conv2D/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/Conv2D/ReadVariableOp^layer7/BiasAdd/ReadVariableOp^layer7/MatMul/ReadVariableOp^layer8/BiasAdd/ReadVariableOp^layer8/MatMul/ReadVariableOp^symbol/BiasAdd/ReadVariableOp^symbol/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????dd: : : : : : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/Conv2D/ReadVariableOplayer4/Conv2D/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/Conv2D/ReadVariableOplayer5/Conv2D/ReadVariableOp2>
layer7/BiasAdd/ReadVariableOplayer7/BiasAdd/ReadVariableOp2<
layer7/MatMul/ReadVariableOplayer7/MatMul/ReadVariableOp2>
layer8/BiasAdd/ReadVariableOplayer8/BiasAdd/ReadVariableOp2<
layer8/MatMul/ReadVariableOplayer8/MatMul/ReadVariableOp2>
symbol/BiasAdd/ReadVariableOpsymbol/BiasAdd/ReadVariableOp2<
symbol/MatMul/ReadVariableOpsymbol/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
&__inference_layer1_layer_call_fn_33758

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????__@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_33083w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????__@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
]
A__inference_layer3_layer_call_and_return_conditional_losses_33799

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_layer6_layer_call_fn_33844

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer6_layer_call_and_return_conditional_losses_33147b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????''@:W S
/
_output_shapes
:?????????''@
 
_user_specified_nameinputs
?
]
A__inference_layer6_layer_call_and_return_conditional_losses_33147

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@| ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????''@:W S
/
_output_shapes
:?????????''@
 
_user_specified_nameinputs
?

?
A__inference_symbol_layer_call_and_return_conditional_losses_33910

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_layer8_layer_call_fn_33879

inputs
unknown:
??
	unknown_0:	?
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
GPU 2J 8? *J
fERC
A__inference_layer8_layer_call_and_return_conditional_losses_33177p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
]
A__inference_layer6_layer_call_and_return_conditional_losses_33850

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@| ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????''@:W S
/
_output_shapes
:?????????''@
 
_user_specified_nameinputs
?O
?
__inference__traced_save_34050
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer4_kernel_read_readvariableop*
&savev2_layer4_bias_read_readvariableop,
(savev2_layer5_kernel_read_readvariableop*
&savev2_layer5_bias_read_readvariableop,
(savev2_layer7_kernel_read_readvariableop*
&savev2_layer7_bias_read_readvariableop,
(savev2_layer8_kernel_read_readvariableop*
&savev2_layer8_bias_read_readvariableop,
(savev2_symbol_kernel_read_readvariableop*
&savev2_symbol_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop8
4savev2_rmsprop_layer1_kernel_rms_read_readvariableop6
2savev2_rmsprop_layer1_bias_rms_read_readvariableop8
4savev2_rmsprop_layer2_kernel_rms_read_readvariableop6
2savev2_rmsprop_layer2_bias_rms_read_readvariableop8
4savev2_rmsprop_layer4_kernel_rms_read_readvariableop6
2savev2_rmsprop_layer4_bias_rms_read_readvariableop8
4savev2_rmsprop_layer5_kernel_rms_read_readvariableop6
2savev2_rmsprop_layer5_bias_rms_read_readvariableop8
4savev2_rmsprop_layer7_kernel_rms_read_readvariableop6
2savev2_rmsprop_layer7_bias_rms_read_readvariableop8
4savev2_rmsprop_layer8_kernel_rms_read_readvariableop6
2savev2_rmsprop_layer8_bias_rms_read_readvariableop8
4savev2_rmsprop_symbol_kernel_rms_read_readvariableop6
2savev2_rmsprop_symbol_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer4_kernel_read_readvariableop&savev2_layer4_bias_read_readvariableop(savev2_layer5_kernel_read_readvariableop&savev2_layer5_bias_read_readvariableop(savev2_layer7_kernel_read_readvariableop&savev2_layer7_bias_read_readvariableop(savev2_layer8_kernel_read_readvariableop&savev2_layer8_bias_read_readvariableop(savev2_symbol_kernel_read_readvariableop&savev2_symbol_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop4savev2_rmsprop_layer1_kernel_rms_read_readvariableop2savev2_rmsprop_layer1_bias_rms_read_readvariableop4savev2_rmsprop_layer2_kernel_rms_read_readvariableop2savev2_rmsprop_layer2_bias_rms_read_readvariableop4savev2_rmsprop_layer4_kernel_rms_read_readvariableop2savev2_rmsprop_layer4_bias_rms_read_readvariableop4savev2_rmsprop_layer5_kernel_rms_read_readvariableop2savev2_rmsprop_layer5_bias_rms_read_readvariableop4savev2_rmsprop_layer7_kernel_rms_read_readvariableop2savev2_rmsprop_layer7_bias_rms_read_readvariableop4savev2_rmsprop_layer8_kernel_rms_read_readvariableop2savev2_rmsprop_layer8_bias_rms_read_readvariableop4savev2_rmsprop_symbol_kernel_rms_read_readvariableop2savev2_rmsprop_symbol_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@ : : @:@:@@:@:???:?:
??:?:	?:: : : : : : : : : : : :@:@:@ : : @:@:@@:@:???:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:'	#
!
_output_shapes
:???:!
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
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:, (
&
_output_shapes
:@@: !

_output_shapes
:@:'"#
!
_output_shapes
:???:!#
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
::(

_output_shapes
: 
Ú
?
!__inference__traced_restore_34177
file_prefix8
assignvariableop_layer1_kernel:@,
assignvariableop_1_layer1_bias:@:
 assignvariableop_2_layer2_kernel:@ ,
assignvariableop_3_layer2_bias: :
 assignvariableop_4_layer4_kernel: @,
assignvariableop_5_layer4_bias:@:
 assignvariableop_6_layer5_kernel:@@,
assignvariableop_7_layer5_bias:@5
 assignvariableop_8_layer7_kernel:???-
assignvariableop_9_layer7_bias:	?5
!assignvariableop_10_layer8_kernel:
??.
assignvariableop_11_layer8_bias:	?4
!assignvariableop_12_symbol_kernel:	?-
assignvariableop_13_symbol_bias:*
 assignvariableop_14_rmsprop_iter:	 +
!assignvariableop_15_rmsprop_decay: 3
)assignvariableop_16_rmsprop_learning_rate: .
$assignvariableop_17_rmsprop_momentum: )
assignvariableop_18_rmsprop_rho: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: %
assignvariableop_23_total_2: %
assignvariableop_24_count_2: G
-assignvariableop_25_rmsprop_layer1_kernel_rms:@9
+assignvariableop_26_rmsprop_layer1_bias_rms:@G
-assignvariableop_27_rmsprop_layer2_kernel_rms:@ 9
+assignvariableop_28_rmsprop_layer2_bias_rms: G
-assignvariableop_29_rmsprop_layer4_kernel_rms: @9
+assignvariableop_30_rmsprop_layer4_bias_rms:@G
-assignvariableop_31_rmsprop_layer5_kernel_rms:@@9
+assignvariableop_32_rmsprop_layer5_bias_rms:@B
-assignvariableop_33_rmsprop_layer7_kernel_rms:???:
+assignvariableop_34_rmsprop_layer7_bias_rms:	?A
-assignvariableop_35_rmsprop_layer8_kernel_rms:
??:
+assignvariableop_36_rmsprop_layer8_bias_rms:	?@
-assignvariableop_37_rmsprop_symbol_kernel_rms:	?9
+assignvariableop_38_rmsprop_symbol_bias_rms:
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer7_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer7_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_layer8_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_layer8_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_symbol_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_symbol_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp assignvariableop_14_rmsprop_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_rmsprop_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp)assignvariableop_16_rmsprop_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_rmsprop_momentumIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_rmsprop_rhoIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_rmsprop_layer1_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_rmsprop_layer1_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp-assignvariableop_27_rmsprop_layer2_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_rmsprop_layer2_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp-assignvariableop_29_rmsprop_layer4_kernel_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_rmsprop_layer4_bias_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_rmsprop_layer5_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_rmsprop_layer5_bias_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp-assignvariableop_33_rmsprop_layer7_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_rmsprop_layer7_bias_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp-assignvariableop_35_rmsprop_layer8_kernel_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_rmsprop_layer8_bias_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp-assignvariableop_37_rmsprop_symbol_kernel_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp+assignvariableop_38_rmsprop_symbol_bias_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_38AssignVariableOp_382(
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
?

?
A__inference_layer7_layer_call_and_return_conditional_losses_33870

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_layer1_layer_call_and_return_conditional_losses_33083

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????__@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????__@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????__@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
A__inference_layer5_layer_call_and_return_conditional_losses_33135

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????''@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????''@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????''@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????**@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????**@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????dd:
symbol0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
?

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Xiter
	Ydecay
Zlearning_rate
[momentum
\rho
rms?
rms?
rms?
rms?
*rms?
+rms?
2rms?
3rms?
@rms?
Arms?
Hrms?
Irms?
Prms?
Qrms?"
	optimizer
?
0
1
2
3
*4
+5
26
37
@8
A9
H10
I11
P12
Q13"
trackable_list_wrapper
?
0
1
2
3
*4
+5
26
37
@8
A9
H10
I11
P12
Q13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_Logan_layer_call_fn_33232
%__inference_Logan_layer_call_fn_33569
%__inference_Logan_layer_call_fn_33602
%__inference_Logan_layer_call_fn_33448?
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
?2?
@__inference_Logan_layer_call_and_return_conditional_losses_33658
@__inference_Logan_layer_call_and_return_conditional_losses_33714
@__inference_Logan_layer_call_and_return_conditional_losses_33489
@__inference_Logan_layer_call_and_return_conditional_losses_33530?
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
?B?
 __inference__wrapped_model_33053input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
bserving_default"
signature_map
':%@2layer1/kernel
:@2layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer1_layer_call_fn_33758?
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
A__inference_layer1_layer_call_and_return_conditional_losses_33769?
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
':%@ 2layer2/kernel
: 2layer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer2_layer_call_fn_33778?
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
A__inference_layer2_layer_call_and_return_conditional_losses_33789?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer3_layer_call_fn_33794?
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
A__inference_layer3_layer_call_and_return_conditional_losses_33799?
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
':% @2layer4/kernel
:@2layer4/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer4_layer_call_fn_33808?
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
A__inference_layer4_layer_call_and_return_conditional_losses_33819?
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
':%@@2layer5/kernel
:@2layer5/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer5_layer_call_fn_33828?
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
A__inference_layer5_layer_call_and_return_conditional_losses_33839?
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer6_layer_call_fn_33844?
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
A__inference_layer6_layer_call_and_return_conditional_losses_33850?
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
": ???2layer7/kernel
:?2layer7/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer7_layer_call_fn_33859?
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
A__inference_layer7_layer_call_and_return_conditional_losses_33870?
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
!:
??2layer8/kernel
:?2layer8/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer8_layer_call_fn_33879?
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
A__inference_layer8_layer_call_and_return_conditional_losses_33890?
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
 :	?2symbol/kernel
:2symbol/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_symbol_layer_call_fn_33899?
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
A__inference_symbol_layer_call_and_return_conditional_losses_33910?
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
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
f
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
9"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_signature_wrapper_33749input_1"?
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
 
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
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
1:/@2RMSprop/layer1/kernel/rms
#:!@2RMSprop/layer1/bias/rms
1:/@ 2RMSprop/layer2/kernel/rms
#:! 2RMSprop/layer2/bias/rms
1:/ @2RMSprop/layer4/kernel/rms
#:!@2RMSprop/layer4/bias/rms
1:/@@2RMSprop/layer5/kernel/rms
#:!@2RMSprop/layer5/bias/rms
,:*???2RMSprop/layer7/kernel/rms
$:"?2RMSprop/layer7/bias/rms
+:)
??2RMSprop/layer8/kernel/rms
$:"?2RMSprop/layer8/bias/rms
*:(	?2RMSprop/symbol/kernel/rms
#:!2RMSprop/symbol/bias/rms?
@__inference_Logan_layer_call_and_return_conditional_losses_33489y*+23@AHIPQ@?=
6?3
)?&
input_1?????????dd
p 

 
? "%?"
?
0?????????
? ?
@__inference_Logan_layer_call_and_return_conditional_losses_33530y*+23@AHIPQ@?=
6?3
)?&
input_1?????????dd
p

 
? "%?"
?
0?????????
? ?
@__inference_Logan_layer_call_and_return_conditional_losses_33658x*+23@AHIPQ??<
5?2
(?%
inputs?????????dd
p 

 
? "%?"
?
0?????????
? ?
@__inference_Logan_layer_call_and_return_conditional_losses_33714x*+23@AHIPQ??<
5?2
(?%
inputs?????????dd
p

 
? "%?"
?
0?????????
? ?
%__inference_Logan_layer_call_fn_33232l*+23@AHIPQ@?=
6?3
)?&
input_1?????????dd
p 

 
? "???????????
%__inference_Logan_layer_call_fn_33448l*+23@AHIPQ@?=
6?3
)?&
input_1?????????dd
p

 
? "???????????
%__inference_Logan_layer_call_fn_33569k*+23@AHIPQ??<
5?2
(?%
inputs?????????dd
p 

 
? "???????????
%__inference_Logan_layer_call_fn_33602k*+23@AHIPQ??<
5?2
(?%
inputs?????????dd
p

 
? "???????????
 __inference__wrapped_model_33053{*+23@AHIPQ8?5
.?+
)?&
input_1?????????dd
? "/?,
*
symbol ?
symbol??????????
A__inference_layer1_layer_call_and_return_conditional_losses_33769l7?4
-?*
(?%
inputs?????????dd
? "-?*
#? 
0?????????__@
? ?
&__inference_layer1_layer_call_fn_33758_7?4
-?*
(?%
inputs?????????dd
? " ??????????__@?
A__inference_layer2_layer_call_and_return_conditional_losses_33789l7?4
-?*
(?%
inputs?????????__@
? "-?*
#? 
0?????????ZZ 
? ?
&__inference_layer2_layer_call_fn_33778_7?4
-?*
(?%
inputs?????????__@
? " ??????????ZZ ?
A__inference_layer3_layer_call_and_return_conditional_losses_33799?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_layer3_layer_call_fn_33794?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_layer4_layer_call_and_return_conditional_losses_33819l*+7?4
-?*
(?%
inputs?????????-- 
? "-?*
#? 
0?????????**@
? ?
&__inference_layer4_layer_call_fn_33808_*+7?4
-?*
(?%
inputs?????????-- 
? " ??????????**@?
A__inference_layer5_layer_call_and_return_conditional_losses_33839l237?4
-?*
(?%
inputs?????????**@
? "-?*
#? 
0?????????''@
? ?
&__inference_layer5_layer_call_fn_33828_237?4
-?*
(?%
inputs?????????**@
? " ??????????''@?
A__inference_layer6_layer_call_and_return_conditional_losses_33850b7?4
-?*
(?%
inputs?????????''@
? "'?$
?
0???????????
? 
&__inference_layer6_layer_call_fn_33844U7?4
-?*
(?%
inputs?????????''@
? "?????????????
A__inference_layer7_layer_call_and_return_conditional_losses_33870_@A1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? |
&__inference_layer7_layer_call_fn_33859R@A1?.
'?$
"?
inputs???????????
? "????????????
A__inference_layer8_layer_call_and_return_conditional_losses_33890^HI0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_layer8_layer_call_fn_33879QHI0?-
&?#
!?
inputs??????????
? "????????????
#__inference_signature_wrapper_33749?*+23@AHIPQC?@
? 
9?6
4
input_1)?&
input_1?????????dd"/?,
*
symbol ?
symbol??????????
A__inference_symbol_layer_call_and_return_conditional_losses_33910]PQ0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_symbol_layer_call_fn_33899PPQ0?-
&?#
!?
inputs??????????
? "??????????