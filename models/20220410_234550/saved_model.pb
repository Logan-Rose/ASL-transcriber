??
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
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
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
y
layer7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_namelayer7/kernel
r
!layer7/kernel/Read/ReadVariableOpReadVariableOplayer7/kernel*!
_output_shapes
:???*
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
shape:**
shared_nameRMSprop/layer1/kernel/rms
?
-RMSprop/layer1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer1/kernel/rms*&
_output_shapes
:*
dtype0
?
RMSprop/layer1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameRMSprop/layer1/bias/rms

+RMSprop/layer1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer1/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/layer2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/layer2/kernel/rms
?
-RMSprop/layer2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer2/kernel/rms*&
_output_shapes
:*
dtype0
?
RMSprop/layer2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameRMSprop/layer2/bias/rms

+RMSprop/layer2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer2/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/layer7/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:???**
shared_nameRMSprop/layer7/kernel/rms
?
-RMSprop/layer7/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/layer7/kernel/rms*!
_output_shapes
:???*
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
?A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?A
value?AB?A B?A
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
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
?

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*
?

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
?

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses*
?
Fiter
	Gdecay
Hlearning_rate
Imomentum
Jrho
rms?
rms?
rms?
rms?
.rms?
/rms?
6rms?
7rms?
>rms?
?rms?*
J
0
1
2
3
.4
/5
66
77
>8
?9*
J
0
1
2
3
.4
/5
66
77
>8
?9*
* 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Pserving_default* 
]W
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUElayer7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
?
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUElayer8/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer8/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 
?
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEsymbol/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEsymbol/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

>0
?1*

>0
?1*
* 
?
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*
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
<
0
1
2
3
4
5
6
7*

t0
u1
v2*
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
8
	wtotal
	xcount
y	variables
z	keras_api*
H
	{total
	|count
}
_fn_kwargs
~	variables
	keras_api*
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

w0
x1*

y	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

{0
|1*

~	variables*
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
VARIABLE_VALUERMSprop/layer7/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/layer7/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/layer8/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/layer8/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/symbol/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUERMSprop/symbol/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_7Placeholder*/
_output_shapes
:?????????dd*
dtype0*$
shape:?????????dd
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7layer1/kernellayer1/biaslayer2/kernellayer2/biaslayer7/kernellayer7/biaslayer8/kernellayer8/biassymbol/kernelsymbol/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_133326
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer7/kernel/Read/ReadVariableOplayer7/bias/Read/ReadVariableOp!layer8/kernel/Read/ReadVariableOplayer8/bias/Read/ReadVariableOp!symbol/kernel/Read/ReadVariableOpsymbol/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp-RMSprop/layer1/kernel/rms/Read/ReadVariableOp+RMSprop/layer1/bias/rms/Read/ReadVariableOp-RMSprop/layer2/kernel/rms/Read/ReadVariableOp+RMSprop/layer2/bias/rms/Read/ReadVariableOp-RMSprop/layer7/kernel/rms/Read/ReadVariableOp+RMSprop/layer7/bias/rms/Read/ReadVariableOp-RMSprop/layer8/kernel/rms/Read/ReadVariableOp+RMSprop/layer8/bias/rms/Read/ReadVariableOp-RMSprop/symbol/kernel/rms/Read/ReadVariableOp+RMSprop/symbol/bias/rms/Read/ReadVariableOpConst*,
Tin%
#2!	*
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
__inference__traced_save_133563
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer7/kernellayer7/biaslayer8/kernellayer8/biassymbol/kernelsymbol/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1total_2count_2RMSprop/layer1/kernel/rmsRMSprop/layer1/bias/rmsRMSprop/layer2/kernel/rmsRMSprop/layer2/bias/rmsRMSprop/layer7/kernel/rmsRMSprop/layer7/bias/rmsRMSprop/layer8/kernel/rmsRMSprop/layer8/bias/rmsRMSprop/symbol/kernel/rmsRMSprop/symbol/bias/rms*+
Tin$
"2 *
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
"__inference__traced_restore_133666??
?/
?
A__inference_Logan_layer_call_and_return_conditional_losses_133257

inputs?
%layer1_conv2d_readvariableop_resource:4
&layer1_biasadd_readvariableop_resource:?
%layer2_conv2d_readvariableop_resource:4
&layer2_biasadd_readvariableop_resource::
%layer7_matmul_readvariableop_resource:???5
&layer7_biasadd_readvariableop_resource:	?9
%layer8_matmul_readvariableop_resource:
??5
&layer8_biasadd_readvariableop_resource:	?8
%symbol_matmul_readvariableop_resource:	?4
&symbol_biasadd_readvariableop_resource:
identity??layer1/BiasAdd/ReadVariableOp?layer1/Conv2D/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/Conv2D/ReadVariableOp?layer7/BiasAdd/ReadVariableOp?layer7/MatMul/ReadVariableOp?layer8/BiasAdd/ReadVariableOp?layer8/MatMul/ReadVariableOp?symbol/BiasAdd/ReadVariableOp?symbol/MatMul/ReadVariableOp?
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa*
paddingVALID*
strides
?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aaf
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????aa?
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
layer2/Conv2DConv2Dlayer1/Relu:activations:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^*
paddingVALID*
strides
?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^f
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^^?
layer3/MaxPoolMaxPoollayer2/Relu:activations:0*/
_output_shapes
:?????????//*
ksize
*
paddingVALID*
strides
]
layer6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  }
layer6/ReshapeReshapelayer3/MaxPool:output:0layer6/Const:output:0*
T0*)
_output_shapes
:????????????
layer7/MatMul/ReadVariableOpReadVariableOp%layer7_matmul_readvariableop_resource*!
_output_shapes
:???*
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
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer7/BiasAdd/ReadVariableOp^layer7/MatMul/ReadVariableOp^layer8/BiasAdd/ReadVariableOp^layer8/MatMul/ReadVariableOp^symbol/BiasAdd/ReadVariableOp^symbol/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
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
?4
?
!__inference__wrapped_model_132798
input_7E
+logan_layer1_conv2d_readvariableop_resource::
,logan_layer1_biasadd_readvariableop_resource:E
+logan_layer2_conv2d_readvariableop_resource::
,logan_layer2_biasadd_readvariableop_resource:@
+logan_layer7_matmul_readvariableop_resource:???;
,logan_layer7_biasadd_readvariableop_resource:	??
+logan_layer8_matmul_readvariableop_resource:
??;
,logan_layer8_biasadd_readvariableop_resource:	?>
+logan_symbol_matmul_readvariableop_resource:	?:
,logan_symbol_biasadd_readvariableop_resource:
identity??#Logan/layer1/BiasAdd/ReadVariableOp?"Logan/layer1/Conv2D/ReadVariableOp?#Logan/layer2/BiasAdd/ReadVariableOp?"Logan/layer2/Conv2D/ReadVariableOp?#Logan/layer7/BiasAdd/ReadVariableOp?"Logan/layer7/MatMul/ReadVariableOp?#Logan/layer8/BiasAdd/ReadVariableOp?"Logan/layer8/MatMul/ReadVariableOp?#Logan/symbol/BiasAdd/ReadVariableOp?"Logan/symbol/MatMul/ReadVariableOp?
"Logan/layer1/Conv2D/ReadVariableOpReadVariableOp+logan_layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Logan/layer1/Conv2DConv2Dinput_7*Logan/layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa*
paddingVALID*
strides
?
#Logan/layer1/BiasAdd/ReadVariableOpReadVariableOp,logan_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Logan/layer1/BiasAddBiasAddLogan/layer1/Conv2D:output:0+Logan/layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aar
Logan/layer1/ReluReluLogan/layer1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????aa?
"Logan/layer2/Conv2D/ReadVariableOpReadVariableOp+logan_layer2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Logan/layer2/Conv2DConv2DLogan/layer1/Relu:activations:0*Logan/layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^*
paddingVALID*
strides
?
#Logan/layer2/BiasAdd/ReadVariableOpReadVariableOp,logan_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Logan/layer2/BiasAddBiasAddLogan/layer2/Conv2D:output:0+Logan/layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^r
Logan/layer2/ReluReluLogan/layer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^^?
Logan/layer3/MaxPoolMaxPoolLogan/layer2/Relu:activations:0*/
_output_shapes
:?????????//*
ksize
*
paddingVALID*
strides
c
Logan/layer6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
Logan/layer6/ReshapeReshapeLogan/layer3/MaxPool:output:0Logan/layer6/Const:output:0*
T0*)
_output_shapes
:????????????
"Logan/layer7/MatMul/ReadVariableOpReadVariableOp+logan_layer7_matmul_readvariableop_resource*!
_output_shapes
:???*
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
:??????????
NoOpNoOp$^Logan/layer1/BiasAdd/ReadVariableOp#^Logan/layer1/Conv2D/ReadVariableOp$^Logan/layer2/BiasAdd/ReadVariableOp#^Logan/layer2/Conv2D/ReadVariableOp$^Logan/layer7/BiasAdd/ReadVariableOp#^Logan/layer7/MatMul/ReadVariableOp$^Logan/layer8/BiasAdd/ReadVariableOp#^Logan/layer8/MatMul/ReadVariableOp$^Logan/symbol/BiasAdd/ReadVariableOp#^Logan/symbol/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 2J
#Logan/layer1/BiasAdd/ReadVariableOp#Logan/layer1/BiasAdd/ReadVariableOp2H
"Logan/layer1/Conv2D/ReadVariableOp"Logan/layer1/Conv2D/ReadVariableOp2J
#Logan/layer2/BiasAdd/ReadVariableOp#Logan/layer2/BiasAdd/ReadVariableOp2H
"Logan/layer2/Conv2D/ReadVariableOp"Logan/layer2/Conv2D/ReadVariableOp2J
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
_user_specified_name	input_7
? 
?
A__inference_Logan_layer_call_and_return_conditional_losses_132912

inputs'
layer1_132829:
layer1_132831:'
layer2_132846:
layer2_132848:"
layer7_132872:???
layer7_132874:	?!
layer8_132889:
??
layer8_132891:	? 
symbol_132906:	?
symbol_132908:
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer7/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_132829layer1_132831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_132828?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_132846layer2_132848*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_132845?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_132807?
layer6/PartitionedCallPartitionedCalllayer3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_132858?
layer7/StatefulPartitionedCallStatefulPartitionedCalllayer6/PartitionedCall:output:0layer7_132872layer7_132874*
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
GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_132871?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_132889layer8_132891*
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
GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_132888?
symbol/StatefulPartitionedCallStatefulPartitionedCall'layer8/StatefulPartitionedCall:output:0symbol_132906symbol_132908*
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
B__inference_symbol_layer_call_and_return_conditional_losses_132905v
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer7/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
B__inference_layer2_layer_call_and_return_conditional_losses_132845

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????^^i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????^^w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????aa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????aa
 
_user_specified_nameinputs
?
?
B__inference_layer1_layer_call_and_return_conditional_losses_132828

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aaX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????aai
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????aaw
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
?
?
'__inference_layer7_layer_call_fn_133396

inputs
unknown:???
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
GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_132871p
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
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_layer7_layer_call_and_return_conditional_losses_133407

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
&__inference_Logan_layer_call_fn_133215

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:???
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Logan_layer_call_and_return_conditional_losses_133049o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?
?
B__inference_layer1_layer_call_and_return_conditional_losses_133346

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aaX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????aai
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????aaw
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
?{
?
"__inference__traced_restore_133666
file_prefix8
assignvariableop_layer1_kernel:,
assignvariableop_1_layer1_bias::
 assignvariableop_2_layer2_kernel:,
assignvariableop_3_layer2_bias:5
 assignvariableop_4_layer7_kernel:???-
assignvariableop_5_layer7_bias:	?4
 assignvariableop_6_layer8_kernel:
??-
assignvariableop_7_layer8_bias:	?3
 assignvariableop_8_symbol_kernel:	?,
assignvariableop_9_symbol_bias:*
 assignvariableop_10_rmsprop_iter:	 +
!assignvariableop_11_rmsprop_decay: 3
)assignvariableop_12_rmsprop_learning_rate: .
$assignvariableop_13_rmsprop_momentum: )
assignvariableop_14_rmsprop_rho: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: %
assignvariableop_19_total_2: %
assignvariableop_20_count_2: G
-assignvariableop_21_rmsprop_layer1_kernel_rms:9
+assignvariableop_22_rmsprop_layer1_bias_rms:G
-assignvariableop_23_rmsprop_layer2_kernel_rms:9
+assignvariableop_24_rmsprop_layer2_bias_rms:B
-assignvariableop_25_rmsprop_layer7_kernel_rms:???:
+assignvariableop_26_rmsprop_layer7_bias_rms:	?A
-assignvariableop_27_rmsprop_layer8_kernel_rms:
??:
+assignvariableop_28_rmsprop_layer8_bias_rms:	?@
-assignvariableop_29_rmsprop_symbol_kernel_rms:	?9
+assignvariableop_30_rmsprop_symbol_bias_rms:
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
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
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer8_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer8_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_symbol_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_symbol_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp assignvariableop_10_rmsprop_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_rmsprop_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_rmsprop_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_rmsprop_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_rmsprop_rhoIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp-assignvariableop_21_rmsprop_layer1_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_rmsprop_layer1_bias_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp-assignvariableop_23_rmsprop_layer2_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_rmsprop_layer2_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_rmsprop_layer7_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_rmsprop_layer7_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp-assignvariableop_27_rmsprop_layer8_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_rmsprop_layer8_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp-assignvariableop_29_rmsprop_symbol_kernel_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_rmsprop_symbol_bias_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_30AssignVariableOp_302(
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
B__inference_symbol_layer_call_and_return_conditional_losses_133447

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
'__inference_layer2_layer_call_fn_133355

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_132845w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????^^`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????aa: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????aa
 
_user_specified_nameinputs
?
^
B__inference_layer6_layer_call_and_return_conditional_losses_132858

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????//:W S
/
_output_shapes
:?????????//
 
_user_specified_nameinputs
? 
?
A__inference_Logan_layer_call_and_return_conditional_losses_133049

inputs'
layer1_133021:
layer1_133023:'
layer2_133026:
layer2_133028:"
layer7_133033:???
layer7_133035:	?!
layer8_133038:
??
layer8_133040:	? 
symbol_133043:	?
symbol_133045:
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer7/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_133021layer1_133023*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_132828?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_133026layer2_133028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_132845?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_132807?
layer6/PartitionedCallPartitionedCalllayer3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_132858?
layer7/StatefulPartitionedCallStatefulPartitionedCalllayer6/PartitionedCall:output:0layer7_133033layer7_133035*
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
GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_132871?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_133038layer8_133040*
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
GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_132888?
symbol/StatefulPartitionedCallStatefulPartitionedCall'layer8/StatefulPartitionedCall:output:0symbol_133043symbol_133045*
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
B__inference_symbol_layer_call_and_return_conditional_losses_132905v
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer7/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?

?
B__inference_layer8_layer_call_and_return_conditional_losses_132888

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

?
&__inference_Logan_layer_call_fn_133097
input_7!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:???
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Logan_layer_call_and_return_conditional_losses_133049o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_7
?
C
'__inference_layer6_layer_call_fn_133381

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
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_132858b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????//:W S
/
_output_shapes
:?????????//
 
_user_specified_nameinputs
?/
?
A__inference_Logan_layer_call_and_return_conditional_losses_133299

inputs?
%layer1_conv2d_readvariableop_resource:4
&layer1_biasadd_readvariableop_resource:?
%layer2_conv2d_readvariableop_resource:4
&layer2_biasadd_readvariableop_resource::
%layer7_matmul_readvariableop_resource:???5
&layer7_biasadd_readvariableop_resource:	?9
%layer8_matmul_readvariableop_resource:
??5
&layer8_biasadd_readvariableop_resource:	?8
%symbol_matmul_readvariableop_resource:	?4
&symbol_biasadd_readvariableop_resource:
identity??layer1/BiasAdd/ReadVariableOp?layer1/Conv2D/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/Conv2D/ReadVariableOp?layer7/BiasAdd/ReadVariableOp?layer7/MatMul/ReadVariableOp?layer8/BiasAdd/ReadVariableOp?layer8/MatMul/ReadVariableOp?symbol/BiasAdd/ReadVariableOp?symbol/MatMul/ReadVariableOp?
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aa*
paddingVALID*
strides
?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????aaf
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????aa?
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
layer2/Conv2DConv2Dlayer1/Relu:activations:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^*
paddingVALID*
strides
?
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^f
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^^?
layer3/MaxPoolMaxPoollayer2/Relu:activations:0*/
_output_shapes
:?????????//*
ksize
*
paddingVALID*
strides
]
layer6/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  }
layer6/ReshapeReshapelayer3/MaxPool:output:0layer6/Const:output:0*
T0*)
_output_shapes
:????????????
layer7/MatMul/ReadVariableOpReadVariableOp%layer7_matmul_readvariableop_resource*!
_output_shapes
:???*
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
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer7/BiasAdd/ReadVariableOp^layer7/MatMul/ReadVariableOp^layer8/BiasAdd/ReadVariableOp^layer8/MatMul/ReadVariableOp^symbol/BiasAdd/ReadVariableOp^symbol/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
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
'__inference_layer1_layer_call_fn_133335

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_132828w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????aa`
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
?
^
B__inference_layer6_layer_call_and_return_conditional_losses_133387

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????//:W S
/
_output_shapes
:?????????//
 
_user_specified_nameinputs
?

?
B__inference_symbol_layer_call_and_return_conditional_losses_132905

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
?A
?
__inference__traced_save_133563
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer7_kernel_read_readvariableop&savev2_layer7_bias_read_readvariableop(savev2_layer8_kernel_read_readvariableop&savev2_layer8_bias_read_readvariableop(savev2_symbol_kernel_read_readvariableop&savev2_symbol_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop4savev2_rmsprop_layer1_kernel_rms_read_readvariableop2savev2_rmsprop_layer1_bias_rms_read_readvariableop4savev2_rmsprop_layer2_kernel_rms_read_readvariableop2savev2_rmsprop_layer2_bias_rms_read_readvariableop4savev2_rmsprop_layer7_kernel_rms_read_readvariableop2savev2_rmsprop_layer7_bias_rms_read_readvariableop4savev2_rmsprop_layer8_kernel_rms_read_readvariableop2savev2_rmsprop_layer8_bias_rms_read_readvariableop4savev2_rmsprop_symbol_kernel_rms_read_readvariableop2savev2_rmsprop_symbol_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::???:?:
??:?:	?:: : : : : : : : : : : :::::???:?:
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
::'#
!
_output_shapes
:???:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%	!

_output_shapes
:	?: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::'#
!
_output_shapes
:???:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
:: 

_output_shapes
: 
?

?
B__inference_layer7_layer_call_and_return_conditional_losses_132871

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
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
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
C
'__inference_layer3_layer_call_fn_133371

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
B__inference_layer3_layer_call_and_return_conditional_losses_132807?
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
?
^
B__inference_layer3_layer_call_and_return_conditional_losses_132807

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
^
B__inference_layer3_layer_call_and_return_conditional_losses_133376

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
B__inference_layer8_layer_call_and_return_conditional_losses_133427

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
B__inference_layer2_layer_call_and_return_conditional_losses_133366

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????^^i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????^^w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????aa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????aa
 
_user_specified_nameinputs
? 
?
A__inference_Logan_layer_call_and_return_conditional_losses_133159
input_7'
layer1_133131:
layer1_133133:'
layer2_133136:
layer2_133138:"
layer7_133143:???
layer7_133145:	?!
layer8_133148:
??
layer8_133150:	? 
symbol_133153:	?
symbol_133155:
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer7/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_7layer1_133131layer1_133133*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_132828?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_133136layer2_133138*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_132845?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_132807?
layer6/PartitionedCallPartitionedCalllayer3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_132858?
layer7/StatefulPartitionedCallStatefulPartitionedCalllayer6/PartitionedCall:output:0layer7_133143layer7_133145*
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
GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_132871?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_133148layer8_133150*
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
GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_132888?
symbol/StatefulPartitionedCallStatefulPartitionedCall'layer8/StatefulPartitionedCall:output:0symbol_133153symbol_133155*
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
B__inference_symbol_layer_call_and_return_conditional_losses_132905v
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer7/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_7
?

?
&__inference_Logan_layer_call_fn_133190

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:???
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Logan_layer_call_and_return_conditional_losses_132912o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????dd
 
_user_specified_nameinputs
?

?
&__inference_Logan_layer_call_fn_132935
input_7!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:???
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Logan_layer_call_and_return_conditional_losses_132912o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_7
? 
?
A__inference_Logan_layer_call_and_return_conditional_losses_133128
input_7'
layer1_133100:
layer1_133102:'
layer2_133105:
layer2_133107:"
layer7_133112:???
layer7_133114:	?!
layer8_133117:
??
layer8_133119:	? 
symbol_133122:	?
symbol_133124:
identity??layer1/StatefulPartitionedCall?layer2/StatefulPartitionedCall?layer7/StatefulPartitionedCall?layer8/StatefulPartitionedCall?symbol/StatefulPartitionedCall?
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_7layer1_133100layer1_133102*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????aa*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_132828?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_133105layer2_133107*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_132845?
layer3/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????//* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer3_layer_call_and_return_conditional_losses_132807?
layer6/PartitionedCallPartitionedCalllayer3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer6_layer_call_and_return_conditional_losses_132858?
layer7/StatefulPartitionedCallStatefulPartitionedCalllayer6/PartitionedCall:output:0layer7_133112layer7_133114*
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
GPU 2J 8? *K
fFRD
B__inference_layer7_layer_call_and_return_conditional_losses_132871?
layer8/StatefulPartitionedCallStatefulPartitionedCall'layer7/StatefulPartitionedCall:output:0layer8_133117layer8_133119*
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
GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_132888?
symbol/StatefulPartitionedCallStatefulPartitionedCall'layer8/StatefulPartitionedCall:output:0symbol_133122symbol_133124*
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
B__inference_symbol_layer_call_and_return_conditional_losses_132905v
IdentityIdentity'symbol/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer7/StatefulPartitionedCall^layer8/StatefulPartitionedCall^symbol/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer7/StatefulPartitionedCalllayer7/StatefulPartitionedCall2@
layer8/StatefulPartitionedCalllayer8/StatefulPartitionedCall2@
symbol/StatefulPartitionedCallsymbol/StatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_7
?
?
'__inference_layer8_layer_call_fn_133416

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
GPU 2J 8? *K
fFRD
B__inference_layer8_layer_call_and_return_conditional_losses_132888p
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
?
?
'__inference_symbol_layer_call_fn_133436

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
GPU 2J 8? *K
fFRD
B__inference_symbol_layer_call_and_return_conditional_losses_132905o
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
?

?
$__inference_signature_wrapper_133326
input_7!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:???
	unknown_4:	?
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_132798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????dd
!
_user_specified_name	input_7"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_78
serving_default_input_7:0?????????dd:
symbol0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ڄ
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
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
?

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
?

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Fiter
	Gdecay
Hlearning_rate
Imomentum
Jrho
rms?
rms?
rms?
rms?
.rms?
/rms?
6rms?
7rms?
>rms?
?rms?"
	optimizer
f
0
1
2
3
.4
/5
66
77
>8
?9"
trackable_list_wrapper
f
0
1
2
3
.4
/5
66
77
>8
?9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_Logan_layer_call_fn_132935
&__inference_Logan_layer_call_fn_133190
&__inference_Logan_layer_call_fn_133215
&__inference_Logan_layer_call_fn_133097?
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
A__inference_Logan_layer_call_and_return_conditional_losses_133257
A__inference_Logan_layer_call_and_return_conditional_losses_133299
A__inference_Logan_layer_call_and_return_conditional_losses_133128
A__inference_Logan_layer_call_and_return_conditional_losses_133159?
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
!__inference__wrapped_model_132798input_7"?
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
Pserving_default"
signature_map
':%2layer1/kernel
:2layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer1_layer_call_fn_133335?
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
B__inference_layer1_layer_call_and_return_conditional_losses_133346?
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
':%2layer2/kernel
:2layer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer2_layer_call_fn_133355?
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
B__inference_layer2_layer_call_and_return_conditional_losses_133366?
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
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer3_layer_call_fn_133371?
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
B__inference_layer3_layer_call_and_return_conditional_losses_133376?
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
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer6_layer_call_fn_133381?
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
B__inference_layer6_layer_call_and_return_conditional_losses_133387?
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
": ???2layer7/kernel
:?2layer7/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer7_layer_call_fn_133396?
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
B__inference_layer7_layer_call_and_return_conditional_losses_133407?
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
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer8_layer_call_fn_133416?
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
B__inference_layer8_layer_call_and_return_conditional_losses_133427?
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
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_symbol_layer_call_fn_133436?
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
B__inference_symbol_layer_call_and_return_conditional_losses_133447?
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
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
5
t0
u1
v2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_133326input_7"?
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
N
	wtotal
	xcount
y	variables
z	keras_api"
_tf_keras_metric
^
	{total
	|count
}
_fn_kwargs
~	variables
	keras_api"
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
.
w0
x1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
{0
|1"
trackable_list_wrapper
-
~	variables"
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
1:/2RMSprop/layer1/kernel/rms
#:!2RMSprop/layer1/bias/rms
1:/2RMSprop/layer2/kernel/rms
#:!2RMSprop/layer2/bias/rms
,:*???2RMSprop/layer7/kernel/rms
$:"?2RMSprop/layer7/bias/rms
+:)
??2RMSprop/layer8/kernel/rms
$:"?2RMSprop/layer8/bias/rms
*:(	?2RMSprop/symbol/kernel/rms
#:!2RMSprop/symbol/bias/rms?
A__inference_Logan_layer_call_and_return_conditional_losses_133128u
./67>?@?=
6?3
)?&
input_7?????????dd
p 

 
? "%?"
?
0?????????
? ?
A__inference_Logan_layer_call_and_return_conditional_losses_133159u
./67>?@?=
6?3
)?&
input_7?????????dd
p

 
? "%?"
?
0?????????
? ?
A__inference_Logan_layer_call_and_return_conditional_losses_133257t
./67>???<
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
A__inference_Logan_layer_call_and_return_conditional_losses_133299t
./67>???<
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
&__inference_Logan_layer_call_fn_132935h
./67>?@?=
6?3
)?&
input_7?????????dd
p 

 
? "???????????
&__inference_Logan_layer_call_fn_133097h
./67>?@?=
6?3
)?&
input_7?????????dd
p

 
? "???????????
&__inference_Logan_layer_call_fn_133190g
./67>???<
5?2
(?%
inputs?????????dd
p 

 
? "???????????
&__inference_Logan_layer_call_fn_133215g
./67>???<
5?2
(?%
inputs?????????dd
p

 
? "???????????
!__inference__wrapped_model_132798w
./67>?8?5
.?+
)?&
input_7?????????dd
? "/?,
*
symbol ?
symbol??????????
B__inference_layer1_layer_call_and_return_conditional_losses_133346l7?4
-?*
(?%
inputs?????????dd
? "-?*
#? 
0?????????aa
? ?
'__inference_layer1_layer_call_fn_133335_7?4
-?*
(?%
inputs?????????dd
? " ??????????aa?
B__inference_layer2_layer_call_and_return_conditional_losses_133366l7?4
-?*
(?%
inputs?????????aa
? "-?*
#? 
0?????????^^
? ?
'__inference_layer2_layer_call_fn_133355_7?4
-?*
(?%
inputs?????????aa
? " ??????????^^?
B__inference_layer3_layer_call_and_return_conditional_losses_133376?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_layer3_layer_call_fn_133371?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_layer6_layer_call_and_return_conditional_losses_133387b7?4
-?*
(?%
inputs?????????//
? "'?$
?
0???????????
? ?
'__inference_layer6_layer_call_fn_133381U7?4
-?*
(?%
inputs?????????//
? "?????????????
B__inference_layer7_layer_call_and_return_conditional_losses_133407_./1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? }
'__inference_layer7_layer_call_fn_133396R./1?.
'?$
"?
inputs???????????
? "????????????
B__inference_layer8_layer_call_and_return_conditional_losses_133427^670?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_layer8_layer_call_fn_133416Q670?-
&?#
!?
inputs??????????
? "????????????
$__inference_signature_wrapper_133326?
./67>?C?@
? 
9?6
4
input_7)?&
input_7?????????dd"/?,
*
symbol ?
symbol??????????
B__inference_symbol_layer_call_and_return_conditional_losses_133447]>?0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_symbol_layer_call_fn_133436P>?0?-
&?#
!?
inputs??????????
? "??????????