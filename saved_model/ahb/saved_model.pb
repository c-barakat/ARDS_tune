��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring �
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8�
�
Conv_Layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*$
shared_nameConv_Layer_1/kernel

'Conv_Layer_1/kernel/Read/ReadVariableOpReadVariableOpConv_Layer_1/kernel*"
_output_shapes
:	@*
dtype0
z
Conv_Layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameConv_Layer_1/bias
s
%Conv_Layer_1/bias/Read/ReadVariableOpReadVariableOpConv_Layer_1/bias*
_output_shapes
:@*
dtype0
�
Conv_Layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*$
shared_nameConv_Layer_2/kernel
�
'Conv_Layer_2/kernel/Read/ReadVariableOpReadVariableOpConv_Layer_2/kernel*#
_output_shapes
:@�*
dtype0
{
Conv_Layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameConv_Layer_2/bias
t
%Conv_Layer_2/bias/Read/ReadVariableOpReadVariableOpConv_Layer_2/bias*
_output_shapes	
:�*
dtype0
�
Conv_Layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameConv_Layer_3/kernel
�
'Conv_Layer_3/kernel/Read/ReadVariableOpReadVariableOpConv_Layer_3/kernel*$
_output_shapes
:��*
dtype0
{
Conv_Layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameConv_Layer_3/bias
t
%Conv_Layer_3/bias/Read/ReadVariableOpReadVariableOpConv_Layer_3/bias*
_output_shapes	
:�*
dtype0
�
Conv_Layer_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameConv_Layer_4/kernel
�
'Conv_Layer_4/kernel/Read/ReadVariableOpReadVariableOpConv_Layer_4/kernel*$
_output_shapes
:��*
dtype0
{
Conv_Layer_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameConv_Layer_4/bias
t
%Conv_Layer_4/bias/Read/ReadVariableOpReadVariableOpConv_Layer_4/bias*
_output_shapes	
:�*
dtype0
�
Fully_Connected_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*-
shared_nameFully_Connected_Layer/kernel
�
0Fully_Connected_Layer/kernel/Read/ReadVariableOpReadVariableOpFully_Connected_Layer/kernel*
_output_shapes
:	�*
dtype0
�
Fully_Connected_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameFully_Connected_Layer/bias
�
.Fully_Connected_Layer/bias/Read/ReadVariableOpReadVariableOpFully_Connected_Layer/bias*
_output_shapes
:*
dtype0
�
Output_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameOutput_Layer/kernel
{
'Output_Layer/kernel/Read/ReadVariableOpReadVariableOpOutput_Layer/kernel*
_output_shapes

:*
dtype0
z
Output_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput_Layer/bias
s
%Output_Layer/bias/Read/ReadVariableOpReadVariableOpOutput_Layer/bias*
_output_shapes
:*
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
�
Adam/Conv_Layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*+
shared_nameAdam/Conv_Layer_1/kernel/m
�
.Adam/Conv_Layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_1/kernel/m*"
_output_shapes
:	@*
dtype0
�
Adam/Conv_Layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/Conv_Layer_1/bias/m
�
,Adam/Conv_Layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_1/bias/m*
_output_shapes
:@*
dtype0
�
Adam/Conv_Layer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameAdam/Conv_Layer_2/kernel/m
�
.Adam/Conv_Layer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_2/kernel/m*#
_output_shapes
:@�*
dtype0
�
Adam/Conv_Layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/Conv_Layer_2/bias/m
�
,Adam/Conv_Layer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/Conv_Layer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameAdam/Conv_Layer_3/kernel/m
�
.Adam/Conv_Layer_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_3/kernel/m*$
_output_shapes
:��*
dtype0
�
Adam/Conv_Layer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/Conv_Layer_3/bias/m
�
,Adam/Conv_Layer_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/Conv_Layer_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameAdam/Conv_Layer_4/kernel/m
�
.Adam/Conv_Layer_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_4/kernel/m*$
_output_shapes
:��*
dtype0
�
Adam/Conv_Layer_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/Conv_Layer_4/bias/m
�
,Adam/Conv_Layer_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_4/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/Fully_Connected_Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/Fully_Connected_Layer/kernel/m
�
7Adam/Fully_Connected_Layer/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/Fully_Connected_Layer/kernel/m*
_output_shapes
:	�*
dtype0
�
!Adam/Fully_Connected_Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Fully_Connected_Layer/bias/m
�
5Adam/Fully_Connected_Layer/bias/m/Read/ReadVariableOpReadVariableOp!Adam/Fully_Connected_Layer/bias/m*
_output_shapes
:*
dtype0
�
Adam/Output_Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/Output_Layer/kernel/m
�
.Adam/Output_Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output_Layer/kernel/m*
_output_shapes

:*
dtype0
�
Adam/Output_Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_Layer/bias/m
�
,Adam/Output_Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output_Layer/bias/m*
_output_shapes
:*
dtype0
�
Adam/Conv_Layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*+
shared_nameAdam/Conv_Layer_1/kernel/v
�
.Adam/Conv_Layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_1/kernel/v*"
_output_shapes
:	@*
dtype0
�
Adam/Conv_Layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/Conv_Layer_1/bias/v
�
,Adam/Conv_Layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_1/bias/v*
_output_shapes
:@*
dtype0
�
Adam/Conv_Layer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameAdam/Conv_Layer_2/kernel/v
�
.Adam/Conv_Layer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_2/kernel/v*#
_output_shapes
:@�*
dtype0
�
Adam/Conv_Layer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/Conv_Layer_2/bias/v
�
,Adam/Conv_Layer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/Conv_Layer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameAdam/Conv_Layer_3/kernel/v
�
.Adam/Conv_Layer_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_3/kernel/v*$
_output_shapes
:��*
dtype0
�
Adam/Conv_Layer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/Conv_Layer_3/bias/v
�
,Adam/Conv_Layer_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/Conv_Layer_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameAdam/Conv_Layer_4/kernel/v
�
.Adam/Conv_Layer_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_4/kernel/v*$
_output_shapes
:��*
dtype0
�
Adam/Conv_Layer_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/Conv_Layer_4/bias/v
�
,Adam/Conv_Layer_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_Layer_4/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/Fully_Connected_Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*4
shared_name%#Adam/Fully_Connected_Layer/kernel/v
�
7Adam/Fully_Connected_Layer/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/Fully_Connected_Layer/kernel/v*
_output_shapes
:	�*
dtype0
�
!Adam/Fully_Connected_Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/Fully_Connected_Layer/bias/v
�
5Adam/Fully_Connected_Layer/bias/v/Read/ReadVariableOpReadVariableOp!Adam/Fully_Connected_Layer/bias/v*
_output_shapes
:*
dtype0
�
Adam/Output_Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/Output_Layer/kernel/v
�
.Adam/Output_Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output_Layer/kernel/v*
_output_shapes

:*
dtype0
�
Adam/Output_Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_Layer/bias/v
�
,Adam/Output_Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output_Layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�J
value�JB�J B�J
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
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
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
 	keras_api
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
R
1trainable_variables
2	variables
3regularization_losses
4	keras_api
h

5kernel
6bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
�
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratem�m�m�m�!m�"m�+m�,m�5m�6m�;m�<m�v�v�v�v�!v�"v�+v�,v�5v�6v�;v�<v�
V
0
1
2
3
!4
"5
+6
,7
58
69
;10
<11
V
0
1
2
3
!4
"5
+6
,7
58
69
;10
<11
 
�
Flayer_metrics
trainable_variables
Gnon_trainable_variables
Hmetrics
	variables

Ilayers
regularization_losses
Jlayer_regularization_losses
 
_]
VARIABLE_VALUEConv_Layer_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv_Layer_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Klayer_metrics
trainable_variables
Lnon_trainable_variables
Mmetrics
	variables

Nlayers
regularization_losses
Olayer_regularization_losses
_]
VARIABLE_VALUEConv_Layer_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv_Layer_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Player_metrics
trainable_variables
Qnon_trainable_variables
Rmetrics
	variables

Slayers
regularization_losses
Tlayer_regularization_losses
 
 
 
�
Ulayer_metrics
trainable_variables
Vnon_trainable_variables
Wmetrics
	variables

Xlayers
regularization_losses
Ylayer_regularization_losses
_]
VARIABLE_VALUEConv_Layer_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv_Layer_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
�
Zlayer_metrics
#trainable_variables
[non_trainable_variables
\metrics
$	variables

]layers
%regularization_losses
^layer_regularization_losses
 
 
 
�
_layer_metrics
'trainable_variables
`non_trainable_variables
ametrics
(	variables

blayers
)regularization_losses
clayer_regularization_losses
_]
VARIABLE_VALUEConv_Layer_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEConv_Layer_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
�
dlayer_metrics
-trainable_variables
enon_trainable_variables
fmetrics
.	variables

glayers
/regularization_losses
hlayer_regularization_losses
 
 
 
�
ilayer_metrics
1trainable_variables
jnon_trainable_variables
kmetrics
2	variables

llayers
3regularization_losses
mlayer_regularization_losses
hf
VARIABLE_VALUEFully_Connected_Layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEFully_Connected_Layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61

50
61
 
�
nlayer_metrics
7trainable_variables
onon_trainable_variables
pmetrics
8	variables

qlayers
9regularization_losses
rlayer_regularization_losses
_]
VARIABLE_VALUEOutput_Layer/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEOutput_Layer/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
�
slayer_metrics
=trainable_variables
tnon_trainable_variables
umetrics
>	variables

vlayers
?regularization_losses
wlayer_regularization_losses
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
 

x0
y1
z2
F
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
4
	{total
	|count
}	variables
~	keras_api
H
	total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

}	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
�1

�	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUEAdam/Conv_Layer_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_Layer_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Conv_Layer_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_Layer_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Conv_Layer_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_Layer_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Conv_Layer_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_Layer_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/Fully_Connected_Layer/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/Fully_Connected_Layer/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Output_Layer/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Output_Layer/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Conv_Layer_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_Layer_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Conv_Layer_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_Layer_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Conv_Layer_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_Layer_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Conv_Layer_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_Layer_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/Fully_Connected_Layer/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/Fully_Connected_Layer/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Output_Layer/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Output_Layer/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_Input_LayerPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Input_LayerConv_Layer_1/kernelConv_Layer_1/biasConv_Layer_2/kernelConv_Layer_2/biasConv_Layer_3/kernelConv_Layer_3/biasConv_Layer_4/kernelConv_Layer_4/biasFully_Connected_Layer/kernelFully_Connected_Layer/biasOutput_Layer/kernelOutput_Layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� */
f*R(
&__inference_signature_wrapper_12819404
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'Conv_Layer_1/kernel/Read/ReadVariableOp%Conv_Layer_1/bias/Read/ReadVariableOp'Conv_Layer_2/kernel/Read/ReadVariableOp%Conv_Layer_2/bias/Read/ReadVariableOp'Conv_Layer_3/kernel/Read/ReadVariableOp%Conv_Layer_3/bias/Read/ReadVariableOp'Conv_Layer_4/kernel/Read/ReadVariableOp%Conv_Layer_4/bias/Read/ReadVariableOp0Fully_Connected_Layer/kernel/Read/ReadVariableOp.Fully_Connected_Layer/bias/Read/ReadVariableOp'Output_Layer/kernel/Read/ReadVariableOp%Output_Layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp.Adam/Conv_Layer_1/kernel/m/Read/ReadVariableOp,Adam/Conv_Layer_1/bias/m/Read/ReadVariableOp.Adam/Conv_Layer_2/kernel/m/Read/ReadVariableOp,Adam/Conv_Layer_2/bias/m/Read/ReadVariableOp.Adam/Conv_Layer_3/kernel/m/Read/ReadVariableOp,Adam/Conv_Layer_3/bias/m/Read/ReadVariableOp.Adam/Conv_Layer_4/kernel/m/Read/ReadVariableOp,Adam/Conv_Layer_4/bias/m/Read/ReadVariableOp7Adam/Fully_Connected_Layer/kernel/m/Read/ReadVariableOp5Adam/Fully_Connected_Layer/bias/m/Read/ReadVariableOp.Adam/Output_Layer/kernel/m/Read/ReadVariableOp,Adam/Output_Layer/bias/m/Read/ReadVariableOp.Adam/Conv_Layer_1/kernel/v/Read/ReadVariableOp,Adam/Conv_Layer_1/bias/v/Read/ReadVariableOp.Adam/Conv_Layer_2/kernel/v/Read/ReadVariableOp,Adam/Conv_Layer_2/bias/v/Read/ReadVariableOp.Adam/Conv_Layer_3/kernel/v/Read/ReadVariableOp,Adam/Conv_Layer_3/bias/v/Read/ReadVariableOp.Adam/Conv_Layer_4/kernel/v/Read/ReadVariableOp,Adam/Conv_Layer_4/bias/v/Read/ReadVariableOp7Adam/Fully_Connected_Layer/kernel/v/Read/ReadVariableOp5Adam/Fully_Connected_Layer/bias/v/Read/ReadVariableOp.Adam/Output_Layer/kernel/v/Read/ReadVariableOp,Adam/Output_Layer/bias/v/Read/ReadVariableOpConst*<
Tin5
321	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� **
f%R#
!__inference__traced_save_12820087
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv_Layer_1/kernelConv_Layer_1/biasConv_Layer_2/kernelConv_Layer_2/biasConv_Layer_3/kernelConv_Layer_3/biasConv_Layer_4/kernelConv_Layer_4/biasFully_Connected_Layer/kernelFully_Connected_Layer/biasOutput_Layer/kernelOutput_Layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/Conv_Layer_1/kernel/mAdam/Conv_Layer_1/bias/mAdam/Conv_Layer_2/kernel/mAdam/Conv_Layer_2/bias/mAdam/Conv_Layer_3/kernel/mAdam/Conv_Layer_3/bias/mAdam/Conv_Layer_4/kernel/mAdam/Conv_Layer_4/bias/m#Adam/Fully_Connected_Layer/kernel/m!Adam/Fully_Connected_Layer/bias/mAdam/Output_Layer/kernel/mAdam/Output_Layer/bias/mAdam/Conv_Layer_1/kernel/vAdam/Conv_Layer_1/bias/vAdam/Conv_Layer_2/kernel/vAdam/Conv_Layer_2/bias/vAdam/Conv_Layer_3/kernel/vAdam/Conv_Layer_3/bias/vAdam/Conv_Layer_4/kernel/vAdam/Conv_Layer_4/bias/v#Adam/Fully_Connected_Layer/kernel/v!Adam/Fully_Connected_Layer/bias/vAdam/Output_Layer/kernel/vAdam/Output_Layer/bias/v*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *-
f(R&
$__inference__traced_restore_12820238��
�
k
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_12818986

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�n
�

#__inference__wrapped_model_12818827
input_layer@
<cnn_conv_layer_1_conv1d_expanddims_1_readvariableop_resource4
0cnn_conv_layer_1_biasadd_readvariableop_resource@
<cnn_conv_layer_2_conv1d_expanddims_1_readvariableop_resource4
0cnn_conv_layer_2_biasadd_readvariableop_resource@
<cnn_conv_layer_3_conv1d_expanddims_1_readvariableop_resource4
0cnn_conv_layer_3_biasadd_readvariableop_resource@
<cnn_conv_layer_4_conv1d_expanddims_1_readvariableop_resource4
0cnn_conv_layer_4_biasadd_readvariableop_resource<
8cnn_fully_connected_layer_matmul_readvariableop_resource=
9cnn_fully_connected_layer_biasadd_readvariableop_resource3
/cnn_output_layer_matmul_readvariableop_resource4
0cnn_output_layer_biasadd_readvariableop_resource
identity��'CNN/Conv_Layer_1/BiasAdd/ReadVariableOp�3CNN/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp�'CNN/Conv_Layer_2/BiasAdd/ReadVariableOp�3CNN/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp�'CNN/Conv_Layer_3/BiasAdd/ReadVariableOp�3CNN/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp�'CNN/Conv_Layer_4/BiasAdd/ReadVariableOp�3CNN/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp�0CNN/Fully_Connected_Layer/BiasAdd/ReadVariableOp�/CNN/Fully_Connected_Layer/MatMul/ReadVariableOp�'CNN/Output_Layer/BiasAdd/ReadVariableOp�&CNN/Output_Layer/MatMul/ReadVariableOp�
&CNN/Conv_Layer_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&CNN/Conv_Layer_1/conv1d/ExpandDims/dim�
"CNN/Conv_Layer_1/conv1d/ExpandDims
ExpandDimsinput_layer/CNN/Conv_Layer_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2$
"CNN/Conv_Layer_1/conv1d/ExpandDims�
3CNN/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<cnn_conv_layer_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype025
3CNN/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp�
(CNN/Conv_Layer_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(CNN/Conv_Layer_1/conv1d/ExpandDims_1/dim�
$CNN/Conv_Layer_1/conv1d/ExpandDims_1
ExpandDims;CNN/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp:value:01CNN/Conv_Layer_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2&
$CNN/Conv_Layer_1/conv1d/ExpandDims_1�
CNN/Conv_Layer_1/conv1dConv2D+CNN/Conv_Layer_1/conv1d/ExpandDims:output:0-CNN/Conv_Layer_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
CNN/Conv_Layer_1/conv1d�
CNN/Conv_Layer_1/conv1d/SqueezeSqueeze CNN/Conv_Layer_1/conv1d:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������2!
CNN/Conv_Layer_1/conv1d/Squeeze�
'CNN/Conv_Layer_1/BiasAdd/ReadVariableOpReadVariableOp0cnn_conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'CNN/Conv_Layer_1/BiasAdd/ReadVariableOp�
CNN/Conv_Layer_1/BiasAddBiasAdd(CNN/Conv_Layer_1/conv1d/Squeeze:output:0/CNN/Conv_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@2
CNN/Conv_Layer_1/BiasAdd�
CNN/Conv_Layer_1/ReluRelu!CNN/Conv_Layer_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������@2
CNN/Conv_Layer_1/Relu�
&CNN/Conv_Layer_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&CNN/Conv_Layer_2/conv1d/ExpandDims/dim�
"CNN/Conv_Layer_2/conv1d/ExpandDims
ExpandDims#CNN/Conv_Layer_1/Relu:activations:0/CNN/Conv_Layer_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@2$
"CNN/Conv_Layer_2/conv1d/ExpandDims�
3CNN/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<cnn_conv_layer_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype025
3CNN/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp�
(CNN/Conv_Layer_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(CNN/Conv_Layer_2/conv1d/ExpandDims_1/dim�
$CNN/Conv_Layer_2/conv1d/ExpandDims_1
ExpandDims;CNN/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp:value:01CNN/Conv_Layer_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@�2&
$CNN/Conv_Layer_2/conv1d/ExpandDims_1�
CNN/Conv_Layer_2/conv1dConv2D+CNN/Conv_Layer_2/conv1d/ExpandDims:output:0-CNN/Conv_Layer_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
CNN/Conv_Layer_2/conv1d�
CNN/Conv_Layer_2/conv1d/SqueezeSqueeze CNN/Conv_Layer_2/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2!
CNN/Conv_Layer_2/conv1d/Squeeze�
'CNN/Conv_Layer_2/BiasAdd/ReadVariableOpReadVariableOp0cnn_conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'CNN/Conv_Layer_2/BiasAdd/ReadVariableOp�
CNN/Conv_Layer_2/BiasAddBiasAdd(CNN/Conv_Layer_2/conv1d/Squeeze:output:0/CNN/Conv_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
CNN/Conv_Layer_2/BiasAdd�
CNN/Conv_Layer_2/ReluRelu!CNN/Conv_Layer_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
CNN/Conv_Layer_2/Relu�
CNN/Dropout_Layer_1/IdentityIdentity#CNN/Conv_Layer_2/Relu:activations:0*
T0*,
_output_shapes
:����������2
CNN/Dropout_Layer_1/Identity�
&CNN/Conv_Layer_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&CNN/Conv_Layer_3/conv1d/ExpandDims/dim�
"CNN/Conv_Layer_3/conv1d/ExpandDims
ExpandDims%CNN/Dropout_Layer_1/Identity:output:0/CNN/Conv_Layer_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2$
"CNN/Conv_Layer_3/conv1d/ExpandDims�
3CNN/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<cnn_conv_layer_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype025
3CNN/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp�
(CNN/Conv_Layer_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(CNN/Conv_Layer_3/conv1d/ExpandDims_1/dim�
$CNN/Conv_Layer_3/conv1d/ExpandDims_1
ExpandDims;CNN/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp:value:01CNN/Conv_Layer_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2&
$CNN/Conv_Layer_3/conv1d/ExpandDims_1�
CNN/Conv_Layer_3/conv1dConv2D+CNN/Conv_Layer_3/conv1d/ExpandDims:output:0-CNN/Conv_Layer_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
CNN/Conv_Layer_3/conv1d�
CNN/Conv_Layer_3/conv1d/SqueezeSqueeze CNN/Conv_Layer_3/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2!
CNN/Conv_Layer_3/conv1d/Squeeze�
'CNN/Conv_Layer_3/BiasAdd/ReadVariableOpReadVariableOp0cnn_conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'CNN/Conv_Layer_3/BiasAdd/ReadVariableOp�
CNN/Conv_Layer_3/BiasAddBiasAdd(CNN/Conv_Layer_3/conv1d/Squeeze:output:0/CNN/Conv_Layer_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
CNN/Conv_Layer_3/BiasAdd�
CNN/Conv_Layer_3/ReluRelu!CNN/Conv_Layer_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
CNN/Conv_Layer_3/Relu�
CNN/Dropout_Layer_2/IdentityIdentity#CNN/Conv_Layer_3/Relu:activations:0*
T0*,
_output_shapes
:����������2
CNN/Dropout_Layer_2/Identity�
&CNN/Conv_Layer_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&CNN/Conv_Layer_4/conv1d/ExpandDims/dim�
"CNN/Conv_Layer_4/conv1d/ExpandDims
ExpandDims%CNN/Dropout_Layer_2/Identity:output:0/CNN/Conv_Layer_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2$
"CNN/Conv_Layer_4/conv1d/ExpandDims�
3CNN/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<cnn_conv_layer_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype025
3CNN/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp�
(CNN/Conv_Layer_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(CNN/Conv_Layer_4/conv1d/ExpandDims_1/dim�
$CNN/Conv_Layer_4/conv1d/ExpandDims_1
ExpandDims;CNN/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp:value:01CNN/Conv_Layer_4/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2&
$CNN/Conv_Layer_4/conv1d/ExpandDims_1�
CNN/Conv_Layer_4/conv1dConv2D+CNN/Conv_Layer_4/conv1d/ExpandDims:output:0-CNN/Conv_Layer_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
CNN/Conv_Layer_4/conv1d�
CNN/Conv_Layer_4/conv1d/SqueezeSqueeze CNN/Conv_Layer_4/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2!
CNN/Conv_Layer_4/conv1d/Squeeze�
'CNN/Conv_Layer_4/BiasAdd/ReadVariableOpReadVariableOp0cnn_conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'CNN/Conv_Layer_4/BiasAdd/ReadVariableOp�
CNN/Conv_Layer_4/BiasAddBiasAdd(CNN/Conv_Layer_4/conv1d/Squeeze:output:0/CNN/Conv_Layer_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
CNN/Conv_Layer_4/BiasAdd�
CNN/Conv_Layer_4/ReluRelu!CNN/Conv_Layer_4/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
CNN/Conv_Layer_4/Relu�
CNN/Flatten_Layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
CNN/Flatten_Layer/Const�
CNN/Flatten_Layer/ReshapeReshape#CNN/Conv_Layer_4/Relu:activations:0 CNN/Flatten_Layer/Const:output:0*
T0*(
_output_shapes
:����������2
CNN/Flatten_Layer/Reshape�
/CNN/Fully_Connected_Layer/MatMul/ReadVariableOpReadVariableOp8cnn_fully_connected_layer_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype021
/CNN/Fully_Connected_Layer/MatMul/ReadVariableOp�
 CNN/Fully_Connected_Layer/MatMulMatMul"CNN/Flatten_Layer/Reshape:output:07CNN/Fully_Connected_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2"
 CNN/Fully_Connected_Layer/MatMul�
0CNN/Fully_Connected_Layer/BiasAdd/ReadVariableOpReadVariableOp9cnn_fully_connected_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0CNN/Fully_Connected_Layer/BiasAdd/ReadVariableOp�
!CNN/Fully_Connected_Layer/BiasAddBiasAdd*CNN/Fully_Connected_Layer/MatMul:product:08CNN/Fully_Connected_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!CNN/Fully_Connected_Layer/BiasAdd�
CNN/Fully_Connected_Layer/ReluRelu*CNN/Fully_Connected_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2 
CNN/Fully_Connected_Layer/Relu�
&CNN/Output_Layer/MatMul/ReadVariableOpReadVariableOp/cnn_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&CNN/Output_Layer/MatMul/ReadVariableOp�
CNN/Output_Layer/MatMulMatMul,CNN/Fully_Connected_Layer/Relu:activations:0.CNN/Output_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
CNN/Output_Layer/MatMul�
'CNN/Output_Layer/BiasAdd/ReadVariableOpReadVariableOp0cnn_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'CNN/Output_Layer/BiasAdd/ReadVariableOp�
CNN/Output_Layer/BiasAddBiasAdd!CNN/Output_Layer/MatMul:product:0/CNN/Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
CNN/Output_Layer/BiasAdd�
IdentityIdentity!CNN/Output_Layer/BiasAdd:output:0(^CNN/Conv_Layer_1/BiasAdd/ReadVariableOp4^CNN/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp(^CNN/Conv_Layer_2/BiasAdd/ReadVariableOp4^CNN/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp(^CNN/Conv_Layer_3/BiasAdd/ReadVariableOp4^CNN/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp(^CNN/Conv_Layer_4/BiasAdd/ReadVariableOp4^CNN/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp1^CNN/Fully_Connected_Layer/BiasAdd/ReadVariableOp0^CNN/Fully_Connected_Layer/MatMul/ReadVariableOp(^CNN/Output_Layer/BiasAdd/ReadVariableOp'^CNN/Output_Layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::2R
'CNN/Conv_Layer_1/BiasAdd/ReadVariableOp'CNN/Conv_Layer_1/BiasAdd/ReadVariableOp2j
3CNN/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp3CNN/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp2R
'CNN/Conv_Layer_2/BiasAdd/ReadVariableOp'CNN/Conv_Layer_2/BiasAdd/ReadVariableOp2j
3CNN/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp3CNN/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp2R
'CNN/Conv_Layer_3/BiasAdd/ReadVariableOp'CNN/Conv_Layer_3/BiasAdd/ReadVariableOp2j
3CNN/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp3CNN/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp2R
'CNN/Conv_Layer_4/BiasAdd/ReadVariableOp'CNN/Conv_Layer_4/BiasAdd/ReadVariableOp2j
3CNN/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp3CNN/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp2d
0CNN/Fully_Connected_Layer/BiasAdd/ReadVariableOp0CNN/Fully_Connected_Layer/BiasAdd/ReadVariableOp2b
/CNN/Fully_Connected_Layer/MatMul/ReadVariableOp/CNN/Fully_Connected_Layer/MatMul/ReadVariableOp2R
'CNN/Output_Layer/BiasAdd/ReadVariableOp'CNN/Output_Layer/BiasAdd/ReadVariableOp2P
&CNN/Output_Layer/MatMul/ReadVariableOp&CNN/Output_Layer/MatMul/ReadVariableOp:X T
+
_output_shapes
:���������
%
_user_specified_nameInput_Layer
�
�
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_12818953

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp6^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
l
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_12819788

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_12818885

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@�2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp6^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
&__inference_CNN_layer_call_fn_12819650

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

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_CNN_layer_call_and_return_conditional_losses_128193202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_12819666

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_12819923B
>conv_layer_4_kernel_regularizer_square_readvariableop_resource
identity��5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>conv_layer_4_kernel_regularizer_square_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentity'Conv_Layer_4/kernel/Regularizer/mul:z:06^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp
�
k
2__inference_Dropout_Layer_2_layer_call_fn_12819798

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_128189812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_12819043

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
&__inference_CNN_layer_call_fn_12819347
input_layer
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

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_CNN_layer_call_and_return_conditional_losses_128193202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_nameInput_Layer
�	
�
J__inference_Output_Layer_layer_call_and_return_conditional_losses_12819881

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_12819703

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@�2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp6^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
$__inference__traced_restore_12820238
file_prefix(
$assignvariableop_conv_layer_1_kernel(
$assignvariableop_1_conv_layer_1_bias*
&assignvariableop_2_conv_layer_2_kernel(
$assignvariableop_3_conv_layer_2_bias*
&assignvariableop_4_conv_layer_3_kernel(
$assignvariableop_5_conv_layer_3_bias*
&assignvariableop_6_conv_layer_4_kernel(
$assignvariableop_7_conv_layer_4_bias3
/assignvariableop_8_fully_connected_layer_kernel1
-assignvariableop_9_fully_connected_layer_bias+
'assignvariableop_10_output_layer_kernel)
%assignvariableop_11_output_layer_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1
assignvariableop_21_total_2
assignvariableop_22_count_22
.assignvariableop_23_adam_conv_layer_1_kernel_m0
,assignvariableop_24_adam_conv_layer_1_bias_m2
.assignvariableop_25_adam_conv_layer_2_kernel_m0
,assignvariableop_26_adam_conv_layer_2_bias_m2
.assignvariableop_27_adam_conv_layer_3_kernel_m0
,assignvariableop_28_adam_conv_layer_3_bias_m2
.assignvariableop_29_adam_conv_layer_4_kernel_m0
,assignvariableop_30_adam_conv_layer_4_bias_m;
7assignvariableop_31_adam_fully_connected_layer_kernel_m9
5assignvariableop_32_adam_fully_connected_layer_bias_m2
.assignvariableop_33_adam_output_layer_kernel_m0
,assignvariableop_34_adam_output_layer_bias_m2
.assignvariableop_35_adam_conv_layer_1_kernel_v0
,assignvariableop_36_adam_conv_layer_1_bias_v2
.assignvariableop_37_adam_conv_layer_2_kernel_v0
,assignvariableop_38_adam_conv_layer_2_bias_v2
.assignvariableop_39_adam_conv_layer_3_kernel_v0
,assignvariableop_40_adam_conv_layer_3_bias_v2
.assignvariableop_41_adam_conv_layer_4_kernel_v0
,assignvariableop_42_adam_conv_layer_4_bias_v;
7assignvariableop_43_adam_fully_connected_layer_kernel_v9
5assignvariableop_44_adam_fully_connected_layer_bias_v2
.assignvariableop_45_adam_output_layer_kernel_v0
,assignvariableop_46_adam_output_layer_bias_v
identity_48��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*�
value�B�0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp$assignvariableop_conv_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_conv_layer_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv_layer_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_conv_layer_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv_layer_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_conv_layer_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_conv_layer_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_fully_connected_layer_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_fully_connected_layer_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_output_layer_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_output_layer_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_conv_layer_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_conv_layer_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_conv_layer_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_conv_layer_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_conv_layer_3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_conv_layer_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_conv_layer_4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_conv_layer_4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_fully_connected_layer_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_fully_connected_layer_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_output_layer_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_output_layer_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_conv_layer_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv_layer_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_conv_layer_2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_conv_layer_2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_conv_layer_3_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_conv_layer_3_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp.assignvariableop_41_adam_conv_layer_4_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_conv_layer_4_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_fully_connected_layer_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_fully_connected_layer_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp.assignvariableop_45_adam_output_layer_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_output_layer_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_469
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47�
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_46AssignVariableOp_462(
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
�
�
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_12819767

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp6^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_CNN_layer_call_and_return_conditional_losses_12819505

inputs<
8conv_layer_1_conv1d_expanddims_1_readvariableop_resource0
,conv_layer_1_biasadd_readvariableop_resource<
8conv_layer_2_conv1d_expanddims_1_readvariableop_resource0
,conv_layer_2_biasadd_readvariableop_resource<
8conv_layer_3_conv1d_expanddims_1_readvariableop_resource0
,conv_layer_3_biasadd_readvariableop_resource<
8conv_layer_4_conv1d_expanddims_1_readvariableop_resource0
,conv_layer_4_biasadd_readvariableop_resource8
4fully_connected_layer_matmul_readvariableop_resource9
5fully_connected_layer_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��#Conv_Layer_1/BiasAdd/ReadVariableOp�/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp�#Conv_Layer_2/BiasAdd/ReadVariableOp�/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp�5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�#Conv_Layer_3/BiasAdd/ReadVariableOp�/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp�5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�#Conv_Layer_4/BiasAdd/ReadVariableOp�/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp�5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�,Fully_Connected_Layer/BiasAdd/ReadVariableOp�+Fully_Connected_Layer/MatMul/ReadVariableOp�#Output_Layer/BiasAdd/ReadVariableOp�"Output_Layer/MatMul/ReadVariableOp�
"Conv_Layer_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"Conv_Layer_1/conv1d/ExpandDims/dim�
Conv_Layer_1/conv1d/ExpandDims
ExpandDimsinputs+Conv_Layer_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2 
Conv_Layer_1/conv1d/ExpandDims�
/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv_layer_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype021
/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp�
$Conv_Layer_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv_Layer_1/conv1d/ExpandDims_1/dim�
 Conv_Layer_1/conv1d/ExpandDims_1
ExpandDims7Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv_Layer_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2"
 Conv_Layer_1/conv1d/ExpandDims_1�
Conv_Layer_1/conv1dConv2D'Conv_Layer_1/conv1d/ExpandDims:output:0)Conv_Layer_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
Conv_Layer_1/conv1d�
Conv_Layer_1/conv1d/SqueezeSqueezeConv_Layer_1/conv1d:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������2
Conv_Layer_1/conv1d/Squeeze�
#Conv_Layer_1/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#Conv_Layer_1/BiasAdd/ReadVariableOp�
Conv_Layer_1/BiasAddBiasAdd$Conv_Layer_1/conv1d/Squeeze:output:0+Conv_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@2
Conv_Layer_1/BiasAdd�
Conv_Layer_1/ReluReluConv_Layer_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������@2
Conv_Layer_1/Relu�
"Conv_Layer_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"Conv_Layer_2/conv1d/ExpandDims/dim�
Conv_Layer_2/conv1d/ExpandDims
ExpandDimsConv_Layer_1/Relu:activations:0+Conv_Layer_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@2 
Conv_Layer_2/conv1d/ExpandDims�
/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv_layer_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype021
/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp�
$Conv_Layer_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv_Layer_2/conv1d/ExpandDims_1/dim�
 Conv_Layer_2/conv1d/ExpandDims_1
ExpandDims7Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv_Layer_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@�2"
 Conv_Layer_2/conv1d/ExpandDims_1�
Conv_Layer_2/conv1dConv2D'Conv_Layer_2/conv1d/ExpandDims:output:0)Conv_Layer_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv_Layer_2/conv1d�
Conv_Layer_2/conv1d/SqueezeSqueezeConv_Layer_2/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
Conv_Layer_2/conv1d/Squeeze�
#Conv_Layer_2/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#Conv_Layer_2/BiasAdd/ReadVariableOp�
Conv_Layer_2/BiasAddBiasAdd$Conv_Layer_2/conv1d/Squeeze:output:0+Conv_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
Conv_Layer_2/BiasAdd�
Conv_Layer_2/ReluReluConv_Layer_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
Conv_Layer_2/Relu�
Dropout_Layer_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Dropout_Layer_1/dropout/Const�
Dropout_Layer_1/dropout/MulMulConv_Layer_2/Relu:activations:0&Dropout_Layer_1/dropout/Const:output:0*
T0*,
_output_shapes
:����������2
Dropout_Layer_1/dropout/Mul�
Dropout_Layer_1/dropout/ShapeShapeConv_Layer_2/Relu:activations:0*
T0*
_output_shapes
:2
Dropout_Layer_1/dropout/Shape�
4Dropout_Layer_1/dropout/random_uniform/RandomUniformRandomUniform&Dropout_Layer_1/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype026
4Dropout_Layer_1/dropout/random_uniform/RandomUniform�
&Dropout_Layer_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&Dropout_Layer_1/dropout/GreaterEqual/y�
$Dropout_Layer_1/dropout/GreaterEqualGreaterEqual=Dropout_Layer_1/dropout/random_uniform/RandomUniform:output:0/Dropout_Layer_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2&
$Dropout_Layer_1/dropout/GreaterEqual�
Dropout_Layer_1/dropout/CastCast(Dropout_Layer_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
Dropout_Layer_1/dropout/Cast�
Dropout_Layer_1/dropout/Mul_1MulDropout_Layer_1/dropout/Mul:z:0 Dropout_Layer_1/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
Dropout_Layer_1/dropout/Mul_1�
"Conv_Layer_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"Conv_Layer_3/conv1d/ExpandDims/dim�
Conv_Layer_3/conv1d/ExpandDims
ExpandDims!Dropout_Layer_1/dropout/Mul_1:z:0+Conv_Layer_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2 
Conv_Layer_3/conv1d/ExpandDims�
/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv_layer_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype021
/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp�
$Conv_Layer_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv_Layer_3/conv1d/ExpandDims_1/dim�
 Conv_Layer_3/conv1d/ExpandDims_1
ExpandDims7Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv_Layer_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2"
 Conv_Layer_3/conv1d/ExpandDims_1�
Conv_Layer_3/conv1dConv2D'Conv_Layer_3/conv1d/ExpandDims:output:0)Conv_Layer_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv_Layer_3/conv1d�
Conv_Layer_3/conv1d/SqueezeSqueezeConv_Layer_3/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
Conv_Layer_3/conv1d/Squeeze�
#Conv_Layer_3/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#Conv_Layer_3/BiasAdd/ReadVariableOp�
Conv_Layer_3/BiasAddBiasAdd$Conv_Layer_3/conv1d/Squeeze:output:0+Conv_Layer_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
Conv_Layer_3/BiasAdd�
Conv_Layer_3/ReluReluConv_Layer_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
Conv_Layer_3/Relu�
Dropout_Layer_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Dropout_Layer_2/dropout/Const�
Dropout_Layer_2/dropout/MulMulConv_Layer_3/Relu:activations:0&Dropout_Layer_2/dropout/Const:output:0*
T0*,
_output_shapes
:����������2
Dropout_Layer_2/dropout/Mul�
Dropout_Layer_2/dropout/ShapeShapeConv_Layer_3/Relu:activations:0*
T0*
_output_shapes
:2
Dropout_Layer_2/dropout/Shape�
4Dropout_Layer_2/dropout/random_uniform/RandomUniformRandomUniform&Dropout_Layer_2/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype026
4Dropout_Layer_2/dropout/random_uniform/RandomUniform�
&Dropout_Layer_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2(
&Dropout_Layer_2/dropout/GreaterEqual/y�
$Dropout_Layer_2/dropout/GreaterEqualGreaterEqual=Dropout_Layer_2/dropout/random_uniform/RandomUniform:output:0/Dropout_Layer_2/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2&
$Dropout_Layer_2/dropout/GreaterEqual�
Dropout_Layer_2/dropout/CastCast(Dropout_Layer_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
Dropout_Layer_2/dropout/Cast�
Dropout_Layer_2/dropout/Mul_1MulDropout_Layer_2/dropout/Mul:z:0 Dropout_Layer_2/dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
Dropout_Layer_2/dropout/Mul_1�
"Conv_Layer_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"Conv_Layer_4/conv1d/ExpandDims/dim�
Conv_Layer_4/conv1d/ExpandDims
ExpandDims!Dropout_Layer_2/dropout/Mul_1:z:0+Conv_Layer_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2 
Conv_Layer_4/conv1d/ExpandDims�
/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv_layer_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype021
/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp�
$Conv_Layer_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv_Layer_4/conv1d/ExpandDims_1/dim�
 Conv_Layer_4/conv1d/ExpandDims_1
ExpandDims7Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv_Layer_4/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2"
 Conv_Layer_4/conv1d/ExpandDims_1�
Conv_Layer_4/conv1dConv2D'Conv_Layer_4/conv1d/ExpandDims:output:0)Conv_Layer_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv_Layer_4/conv1d�
Conv_Layer_4/conv1d/SqueezeSqueezeConv_Layer_4/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
Conv_Layer_4/conv1d/Squeeze�
#Conv_Layer_4/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#Conv_Layer_4/BiasAdd/ReadVariableOp�
Conv_Layer_4/BiasAddBiasAdd$Conv_Layer_4/conv1d/Squeeze:output:0+Conv_Layer_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
Conv_Layer_4/BiasAdd�
Conv_Layer_4/ReluReluConv_Layer_4/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
Conv_Layer_4/Relu{
Flatten_Layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Flatten_Layer/Const�
Flatten_Layer/ReshapeReshapeConv_Layer_4/Relu:activations:0Flatten_Layer/Const:output:0*
T0*(
_output_shapes
:����������2
Flatten_Layer/Reshape�
+Fully_Connected_Layer/MatMul/ReadVariableOpReadVariableOp4fully_connected_layer_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02-
+Fully_Connected_Layer/MatMul/ReadVariableOp�
Fully_Connected_Layer/MatMulMatMulFlatten_Layer/Reshape:output:03Fully_Connected_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Fully_Connected_Layer/MatMul�
,Fully_Connected_Layer/BiasAdd/ReadVariableOpReadVariableOp5fully_connected_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,Fully_Connected_Layer/BiasAdd/ReadVariableOp�
Fully_Connected_Layer/BiasAddBiasAdd&Fully_Connected_Layer/MatMul:product:04Fully_Connected_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Fully_Connected_Layer/BiasAdd�
Fully_Connected_Layer/ReluRelu&Fully_Connected_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
Fully_Connected_Layer/Relu�
"Output_Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"Output_Layer/MatMul/ReadVariableOp�
Output_Layer/MatMulMatMul(Fully_Connected_Layer/Relu:activations:0*Output_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Output_Layer/MatMul�
#Output_Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Output_Layer/BiasAdd/ReadVariableOp�
Output_Layer/BiasAddBiasAddOutput_Layer/MatMul:product:0+Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Output_Layer/BiasAdd�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_layer_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_layer_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_layer_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentityOutput_Layer/BiasAdd:output:0$^Conv_Layer_1/BiasAdd/ReadVariableOp0^Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp$^Conv_Layer_2/BiasAdd/ReadVariableOp0^Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp6^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp$^Conv_Layer_3/BiasAdd/ReadVariableOp0^Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp6^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp$^Conv_Layer_4/BiasAdd/ReadVariableOp0^Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp6^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp-^Fully_Connected_Layer/BiasAdd/ReadVariableOp,^Fully_Connected_Layer/MatMul/ReadVariableOp$^Output_Layer/BiasAdd/ReadVariableOp#^Output_Layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::2J
#Conv_Layer_1/BiasAdd/ReadVariableOp#Conv_Layer_1/BiasAdd/ReadVariableOp2b
/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv_Layer_2/BiasAdd/ReadVariableOp#Conv_Layer_2/BiasAdd/ReadVariableOp2b
/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp2J
#Conv_Layer_3/BiasAdd/ReadVariableOp#Conv_Layer_3/BiasAdd/ReadVariableOp2b
/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp2J
#Conv_Layer_4/BiasAdd/ReadVariableOp#Conv_Layer_4/BiasAdd/ReadVariableOp2b
/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp2\
,Fully_Connected_Layer/BiasAdd/ReadVariableOp,Fully_Connected_Layer/BiasAdd/ReadVariableOp2Z
+Fully_Connected_Layer/MatMul/ReadVariableOp+Fully_Connected_Layer/MatMul/ReadVariableOp2J
#Output_Layer/BiasAdd/ReadVariableOp#Output_Layer/BiasAdd/ReadVariableOp2H
"Output_Layer/MatMul/ReadVariableOp"Output_Layer/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_12819021

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp6^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_Flatten_Layer_layer_call_fn_12819851

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_128190432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�c
�
!__inference__traced_save_12820087
file_prefix2
.savev2_conv_layer_1_kernel_read_readvariableop0
,savev2_conv_layer_1_bias_read_readvariableop2
.savev2_conv_layer_2_kernel_read_readvariableop0
,savev2_conv_layer_2_bias_read_readvariableop2
.savev2_conv_layer_3_kernel_read_readvariableop0
,savev2_conv_layer_3_bias_read_readvariableop2
.savev2_conv_layer_4_kernel_read_readvariableop0
,savev2_conv_layer_4_bias_read_readvariableop;
7savev2_fully_connected_layer_kernel_read_readvariableop9
5savev2_fully_connected_layer_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
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
"savev2_count_2_read_readvariableop9
5savev2_adam_conv_layer_1_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_1_bias_m_read_readvariableop9
5savev2_adam_conv_layer_2_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_2_bias_m_read_readvariableop9
5savev2_adam_conv_layer_3_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_3_bias_m_read_readvariableop9
5savev2_adam_conv_layer_4_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_4_bias_m_read_readvariableopB
>savev2_adam_fully_connected_layer_kernel_m_read_readvariableop@
<savev2_adam_fully_connected_layer_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop9
5savev2_adam_conv_layer_1_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_1_bias_v_read_readvariableop9
5savev2_adam_conv_layer_2_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_2_bias_v_read_readvariableop9
5savev2_adam_conv_layer_3_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_3_bias_v_read_readvariableop9
5savev2_adam_conv_layer_4_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_4_bias_v_read_readvariableopB
>savev2_adam_fully_connected_layer_kernel_v_read_readvariableop@
<savev2_adam_fully_connected_layer_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*�
value�B�0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_conv_layer_1_kernel_read_readvariableop,savev2_conv_layer_1_bias_read_readvariableop.savev2_conv_layer_2_kernel_read_readvariableop,savev2_conv_layer_2_bias_read_readvariableop.savev2_conv_layer_3_kernel_read_readvariableop,savev2_conv_layer_3_bias_read_readvariableop.savev2_conv_layer_4_kernel_read_readvariableop,savev2_conv_layer_4_bias_read_readvariableop7savev2_fully_connected_layer_kernel_read_readvariableop5savev2_fully_connected_layer_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop5savev2_adam_conv_layer_1_kernel_m_read_readvariableop3savev2_adam_conv_layer_1_bias_m_read_readvariableop5savev2_adam_conv_layer_2_kernel_m_read_readvariableop3savev2_adam_conv_layer_2_bias_m_read_readvariableop5savev2_adam_conv_layer_3_kernel_m_read_readvariableop3savev2_adam_conv_layer_3_bias_m_read_readvariableop5savev2_adam_conv_layer_4_kernel_m_read_readvariableop3savev2_adam_conv_layer_4_bias_m_read_readvariableop>savev2_adam_fully_connected_layer_kernel_m_read_readvariableop<savev2_adam_fully_connected_layer_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop5savev2_adam_conv_layer_1_kernel_v_read_readvariableop3savev2_adam_conv_layer_1_bias_v_read_readvariableop5savev2_adam_conv_layer_2_kernel_v_read_readvariableop3savev2_adam_conv_layer_2_bias_v_read_readvariableop5savev2_adam_conv_layer_3_kernel_v_read_readvariableop3savev2_adam_conv_layer_3_bias_v_read_readvariableop5savev2_adam_conv_layer_4_kernel_v_read_readvariableop3savev2_adam_conv_layer_4_bias_v_read_readvariableop>savev2_adam_fully_connected_layer_kernel_v_read_readvariableop<savev2_adam_fully_connected_layer_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	@:@:@�:�:��:�:��:�:	�:::: : : : : : : : : : : :	@:@:@�:�:��:�:��:�:	�::::	@:@:@�:�:��:�:��:�:	�:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:	@: 

_output_shapes
:@:)%
#
_output_shapes
:@�:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:%	!

_output_shapes
:	�: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:	@: 

_output_shapes
:@:)%
#
_output_shapes
:@�:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:% !

_output_shapes
:	�: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::($$
"
_output_shapes
:	@: %

_output_shapes
:@:)&%
#
_output_shapes
:@�:!'

_output_shapes	
:�:*(&
$
_output_shapes
:��:!)

_output_shapes	
:�:**&
$
_output_shapes
:��:!+

_output_shapes	
:�:%,!

_output_shapes
:	�: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::0

_output_shapes
: 
�
�
/__inference_Conv_Layer_4_layer_call_fn_12819840

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_128190212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_Fully_Connected_Layer_layer_call_fn_12819871

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_128190622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_12819793

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_12819729

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_Dropout_Layer_2_layer_call_fn_12819803

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_128189862
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_12819846

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_12819901B
>conv_layer_2_kernel_regularizer_square_readvariableop_resource
identity��5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>conv_layer_2_kernel_regularizer_square_readvariableop_resource*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
IdentityIdentity'Conv_Layer_2/kernel/Regularizer/mul:z:06^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp
�	
�
&__inference_signature_wrapper_12819404
input_layer
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

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *,
f'R%
#__inference__wrapped_model_128188272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_nameInput_Layer
�Q
�
A__inference_CNN_layer_call_and_return_conditional_losses_12819236

inputs
conv_layer_1_12819184
conv_layer_1_12819186
conv_layer_2_12819189
conv_layer_2_12819191
conv_layer_3_12819195
conv_layer_3_12819197
conv_layer_4_12819201
conv_layer_4_12819203"
fully_connected_layer_12819207"
fully_connected_layer_12819209
output_layer_12819212
output_layer_12819214
identity��$Conv_Layer_1/StatefulPartitionedCall�$Conv_Layer_2/StatefulPartitionedCall�5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�$Conv_Layer_3/StatefulPartitionedCall�5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�$Conv_Layer_4/StatefulPartitionedCall�5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�'Dropout_Layer_1/StatefulPartitionedCall�'Dropout_Layer_2/StatefulPartitionedCall�-Fully_Connected_Layer/StatefulPartitionedCall�$Output_Layer/StatefulPartitionedCall�
$Conv_Layer_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_layer_1_12819184conv_layer_1_12819186*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_128188472&
$Conv_Layer_1/StatefulPartitionedCall�
$Conv_Layer_2/StatefulPartitionedCallStatefulPartitionedCall-Conv_Layer_1/StatefulPartitionedCall:output:0conv_layer_2_12819189conv_layer_2_12819191*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_128188852&
$Conv_Layer_2/StatefulPartitionedCall�
'Dropout_Layer_1/StatefulPartitionedCallStatefulPartitionedCall-Conv_Layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_128189132)
'Dropout_Layer_1/StatefulPartitionedCall�
$Conv_Layer_3/StatefulPartitionedCallStatefulPartitionedCall0Dropout_Layer_1/StatefulPartitionedCall:output:0conv_layer_3_12819195conv_layer_3_12819197*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_128189532&
$Conv_Layer_3/StatefulPartitionedCall�
'Dropout_Layer_2/StatefulPartitionedCallStatefulPartitionedCall-Conv_Layer_3/StatefulPartitionedCall:output:0(^Dropout_Layer_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_128189812)
'Dropout_Layer_2/StatefulPartitionedCall�
$Conv_Layer_4/StatefulPartitionedCallStatefulPartitionedCall0Dropout_Layer_2/StatefulPartitionedCall:output:0conv_layer_4_12819201conv_layer_4_12819203*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_128190212&
$Conv_Layer_4/StatefulPartitionedCall�
Flatten_Layer/PartitionedCallPartitionedCall-Conv_Layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_128190432
Flatten_Layer/PartitionedCall�
-Fully_Connected_Layer/StatefulPartitionedCallStatefulPartitionedCall&Flatten_Layer/PartitionedCall:output:0fully_connected_layer_12819207fully_connected_layer_12819209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_128190622/
-Fully_Connected_Layer/StatefulPartitionedCall�
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall6Fully_Connected_Layer/StatefulPartitionedCall:output:0output_layer_12819212output_layer_12819214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Output_Layer_layer_call_and_return_conditional_losses_128190882&
$Output_Layer/StatefulPartitionedCall�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_2_12819189*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_3_12819195*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_4_12819201*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0%^Conv_Layer_1/StatefulPartitionedCall%^Conv_Layer_2/StatefulPartitionedCall6^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp%^Conv_Layer_3/StatefulPartitionedCall6^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp%^Conv_Layer_4/StatefulPartitionedCall6^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp(^Dropout_Layer_1/StatefulPartitionedCall(^Dropout_Layer_2/StatefulPartitionedCall.^Fully_Connected_Layer/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::2L
$Conv_Layer_1/StatefulPartitionedCall$Conv_Layer_1/StatefulPartitionedCall2L
$Conv_Layer_2/StatefulPartitionedCall$Conv_Layer_2/StatefulPartitionedCall2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$Conv_Layer_3/StatefulPartitionedCall$Conv_Layer_3/StatefulPartitionedCall2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$Conv_Layer_4/StatefulPartitionedCall$Conv_Layer_4/StatefulPartitionedCall2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp2R
'Dropout_Layer_1/StatefulPartitionedCall'Dropout_Layer_1/StatefulPartitionedCall2R
'Dropout_Layer_2/StatefulPartitionedCall'Dropout_Layer_2/StatefulPartitionedCall2^
-Fully_Connected_Layer/StatefulPartitionedCall-Fully_Connected_Layer/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
&__inference_CNN_layer_call_fn_12819263
input_layer
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

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_CNN_layer_call_and_return_conditional_losses_128192362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_nameInput_Layer
�
�
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_12819831

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Relu�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp6^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_12819912B
>conv_layer_3_kernel_regularizer_square_readvariableop_resource
identity��5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>conv_layer_3_kernel_regularizer_square_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
IdentityIdentity'Conv_Layer_3/kernel/Regularizer/mul:z:06^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp
�Q
�
A__inference_CNN_layer_call_and_return_conditional_losses_12819123
input_layer
conv_layer_1_12818858
conv_layer_1_12818860
conv_layer_2_12818896
conv_layer_2_12818898
conv_layer_3_12818964
conv_layer_3_12818966
conv_layer_4_12819032
conv_layer_4_12819034"
fully_connected_layer_12819073"
fully_connected_layer_12819075
output_layer_12819099
output_layer_12819101
identity��$Conv_Layer_1/StatefulPartitionedCall�$Conv_Layer_2/StatefulPartitionedCall�5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�$Conv_Layer_3/StatefulPartitionedCall�5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�$Conv_Layer_4/StatefulPartitionedCall�5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�'Dropout_Layer_1/StatefulPartitionedCall�'Dropout_Layer_2/StatefulPartitionedCall�-Fully_Connected_Layer/StatefulPartitionedCall�$Output_Layer/StatefulPartitionedCall�
$Conv_Layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_layer_1_12818858conv_layer_1_12818860*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_128188472&
$Conv_Layer_1/StatefulPartitionedCall�
$Conv_Layer_2/StatefulPartitionedCallStatefulPartitionedCall-Conv_Layer_1/StatefulPartitionedCall:output:0conv_layer_2_12818896conv_layer_2_12818898*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_128188852&
$Conv_Layer_2/StatefulPartitionedCall�
'Dropout_Layer_1/StatefulPartitionedCallStatefulPartitionedCall-Conv_Layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_128189132)
'Dropout_Layer_1/StatefulPartitionedCall�
$Conv_Layer_3/StatefulPartitionedCallStatefulPartitionedCall0Dropout_Layer_1/StatefulPartitionedCall:output:0conv_layer_3_12818964conv_layer_3_12818966*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_128189532&
$Conv_Layer_3/StatefulPartitionedCall�
'Dropout_Layer_2/StatefulPartitionedCallStatefulPartitionedCall-Conv_Layer_3/StatefulPartitionedCall:output:0(^Dropout_Layer_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_128189812)
'Dropout_Layer_2/StatefulPartitionedCall�
$Conv_Layer_4/StatefulPartitionedCallStatefulPartitionedCall0Dropout_Layer_2/StatefulPartitionedCall:output:0conv_layer_4_12819032conv_layer_4_12819034*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_128190212&
$Conv_Layer_4/StatefulPartitionedCall�
Flatten_Layer/PartitionedCallPartitionedCall-Conv_Layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_128190432
Flatten_Layer/PartitionedCall�
-Fully_Connected_Layer/StatefulPartitionedCallStatefulPartitionedCall&Flatten_Layer/PartitionedCall:output:0fully_connected_layer_12819073fully_connected_layer_12819075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_128190622/
-Fully_Connected_Layer/StatefulPartitionedCall�
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall6Fully_Connected_Layer/StatefulPartitionedCall:output:0output_layer_12819099output_layer_12819101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Output_Layer_layer_call_and_return_conditional_losses_128190882&
$Output_Layer/StatefulPartitionedCall�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_2_12818896*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_3_12818964*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_4_12819032*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0%^Conv_Layer_1/StatefulPartitionedCall%^Conv_Layer_2/StatefulPartitionedCall6^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp%^Conv_Layer_3/StatefulPartitionedCall6^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp%^Conv_Layer_4/StatefulPartitionedCall6^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp(^Dropout_Layer_1/StatefulPartitionedCall(^Dropout_Layer_2/StatefulPartitionedCall.^Fully_Connected_Layer/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::2L
$Conv_Layer_1/StatefulPartitionedCall$Conv_Layer_1/StatefulPartitionedCall2L
$Conv_Layer_2/StatefulPartitionedCall$Conv_Layer_2/StatefulPartitionedCall2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$Conv_Layer_3/StatefulPartitionedCall$Conv_Layer_3/StatefulPartitionedCall2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$Conv_Layer_4/StatefulPartitionedCall$Conv_Layer_4/StatefulPartitionedCall2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp2R
'Dropout_Layer_1/StatefulPartitionedCall'Dropout_Layer_1/StatefulPartitionedCall2R
'Dropout_Layer_2/StatefulPartitionedCall'Dropout_Layer_2/StatefulPartitionedCall2^
-Fully_Connected_Layer/StatefulPartitionedCall-Fully_Connected_Layer/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_nameInput_Layer
�
l
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_12819724

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_Conv_Layer_3_layer_call_fn_12819776

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_128189532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_12819862

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_Dropout_Layer_1_layer_call_fn_12819739

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_128189182
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_Output_Layer_layer_call_fn_12819890

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Output_Layer_layer_call_and_return_conditional_losses_128190882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
&__inference_CNN_layer_call_fn_12819621

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

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_CNN_layer_call_and_return_conditional_losses_128192362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_Conv_Layer_1_layer_call_fn_12819675

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_128188472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_12818847

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_CNN_layer_call_and_return_conditional_losses_12819592

inputs<
8conv_layer_1_conv1d_expanddims_1_readvariableop_resource0
,conv_layer_1_biasadd_readvariableop_resource<
8conv_layer_2_conv1d_expanddims_1_readvariableop_resource0
,conv_layer_2_biasadd_readvariableop_resource<
8conv_layer_3_conv1d_expanddims_1_readvariableop_resource0
,conv_layer_3_biasadd_readvariableop_resource<
8conv_layer_4_conv1d_expanddims_1_readvariableop_resource0
,conv_layer_4_biasadd_readvariableop_resource8
4fully_connected_layer_matmul_readvariableop_resource9
5fully_connected_layer_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��#Conv_Layer_1/BiasAdd/ReadVariableOp�/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp�#Conv_Layer_2/BiasAdd/ReadVariableOp�/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp�5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�#Conv_Layer_3/BiasAdd/ReadVariableOp�/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp�5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�#Conv_Layer_4/BiasAdd/ReadVariableOp�/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp�5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�,Fully_Connected_Layer/BiasAdd/ReadVariableOp�+Fully_Connected_Layer/MatMul/ReadVariableOp�#Output_Layer/BiasAdd/ReadVariableOp�"Output_Layer/MatMul/ReadVariableOp�
"Conv_Layer_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"Conv_Layer_1/conv1d/ExpandDims/dim�
Conv_Layer_1/conv1d/ExpandDims
ExpandDimsinputs+Conv_Layer_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2 
Conv_Layer_1/conv1d/ExpandDims�
/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv_layer_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype021
/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp�
$Conv_Layer_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv_Layer_1/conv1d/ExpandDims_1/dim�
 Conv_Layer_1/conv1d/ExpandDims_1
ExpandDims7Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv_Layer_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2"
 Conv_Layer_1/conv1d/ExpandDims_1�
Conv_Layer_1/conv1dConv2D'Conv_Layer_1/conv1d/ExpandDims:output:0)Conv_Layer_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
Conv_Layer_1/conv1d�
Conv_Layer_1/conv1d/SqueezeSqueezeConv_Layer_1/conv1d:output:0*
T0*+
_output_shapes
:���������@*
squeeze_dims

���������2
Conv_Layer_1/conv1d/Squeeze�
#Conv_Layer_1/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#Conv_Layer_1/BiasAdd/ReadVariableOp�
Conv_Layer_1/BiasAddBiasAdd$Conv_Layer_1/conv1d/Squeeze:output:0+Conv_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@2
Conv_Layer_1/BiasAdd�
Conv_Layer_1/ReluReluConv_Layer_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������@2
Conv_Layer_1/Relu�
"Conv_Layer_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"Conv_Layer_2/conv1d/ExpandDims/dim�
Conv_Layer_2/conv1d/ExpandDims
ExpandDimsConv_Layer_1/Relu:activations:0+Conv_Layer_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������@2 
Conv_Layer_2/conv1d/ExpandDims�
/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv_layer_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype021
/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp�
$Conv_Layer_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv_Layer_2/conv1d/ExpandDims_1/dim�
 Conv_Layer_2/conv1d/ExpandDims_1
ExpandDims7Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv_Layer_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@�2"
 Conv_Layer_2/conv1d/ExpandDims_1�
Conv_Layer_2/conv1dConv2D'Conv_Layer_2/conv1d/ExpandDims:output:0)Conv_Layer_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv_Layer_2/conv1d�
Conv_Layer_2/conv1d/SqueezeSqueezeConv_Layer_2/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
Conv_Layer_2/conv1d/Squeeze�
#Conv_Layer_2/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#Conv_Layer_2/BiasAdd/ReadVariableOp�
Conv_Layer_2/BiasAddBiasAdd$Conv_Layer_2/conv1d/Squeeze:output:0+Conv_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
Conv_Layer_2/BiasAdd�
Conv_Layer_2/ReluReluConv_Layer_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
Conv_Layer_2/Relu�
Dropout_Layer_1/IdentityIdentityConv_Layer_2/Relu:activations:0*
T0*,
_output_shapes
:����������2
Dropout_Layer_1/Identity�
"Conv_Layer_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"Conv_Layer_3/conv1d/ExpandDims/dim�
Conv_Layer_3/conv1d/ExpandDims
ExpandDims!Dropout_Layer_1/Identity:output:0+Conv_Layer_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2 
Conv_Layer_3/conv1d/ExpandDims�
/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv_layer_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype021
/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp�
$Conv_Layer_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv_Layer_3/conv1d/ExpandDims_1/dim�
 Conv_Layer_3/conv1d/ExpandDims_1
ExpandDims7Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv_Layer_3/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2"
 Conv_Layer_3/conv1d/ExpandDims_1�
Conv_Layer_3/conv1dConv2D'Conv_Layer_3/conv1d/ExpandDims:output:0)Conv_Layer_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv_Layer_3/conv1d�
Conv_Layer_3/conv1d/SqueezeSqueezeConv_Layer_3/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
Conv_Layer_3/conv1d/Squeeze�
#Conv_Layer_3/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#Conv_Layer_3/BiasAdd/ReadVariableOp�
Conv_Layer_3/BiasAddBiasAdd$Conv_Layer_3/conv1d/Squeeze:output:0+Conv_Layer_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
Conv_Layer_3/BiasAdd�
Conv_Layer_3/ReluReluConv_Layer_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
Conv_Layer_3/Relu�
Dropout_Layer_2/IdentityIdentityConv_Layer_3/Relu:activations:0*
T0*,
_output_shapes
:����������2
Dropout_Layer_2/Identity�
"Conv_Layer_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"Conv_Layer_4/conv1d/ExpandDims/dim�
Conv_Layer_4/conv1d/ExpandDims
ExpandDims!Dropout_Layer_2/Identity:output:0+Conv_Layer_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2 
Conv_Layer_4/conv1d/ExpandDims�
/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8conv_layer_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype021
/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp�
$Conv_Layer_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Conv_Layer_4/conv1d/ExpandDims_1/dim�
 Conv_Layer_4/conv1d/ExpandDims_1
ExpandDims7Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp:value:0-Conv_Layer_4/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:��2"
 Conv_Layer_4/conv1d/ExpandDims_1�
Conv_Layer_4/conv1dConv2D'Conv_Layer_4/conv1d/ExpandDims:output:0)Conv_Layer_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv_Layer_4/conv1d�
Conv_Layer_4/conv1d/SqueezeSqueezeConv_Layer_4/conv1d:output:0*
T0*,
_output_shapes
:����������*
squeeze_dims

���������2
Conv_Layer_4/conv1d/Squeeze�
#Conv_Layer_4/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#Conv_Layer_4/BiasAdd/ReadVariableOp�
Conv_Layer_4/BiasAddBiasAdd$Conv_Layer_4/conv1d/Squeeze:output:0+Conv_Layer_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
Conv_Layer_4/BiasAdd�
Conv_Layer_4/ReluReluConv_Layer_4/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
Conv_Layer_4/Relu{
Flatten_Layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Flatten_Layer/Const�
Flatten_Layer/ReshapeReshapeConv_Layer_4/Relu:activations:0Flatten_Layer/Const:output:0*
T0*(
_output_shapes
:����������2
Flatten_Layer/Reshape�
+Fully_Connected_Layer/MatMul/ReadVariableOpReadVariableOp4fully_connected_layer_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02-
+Fully_Connected_Layer/MatMul/ReadVariableOp�
Fully_Connected_Layer/MatMulMatMulFlatten_Layer/Reshape:output:03Fully_Connected_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Fully_Connected_Layer/MatMul�
,Fully_Connected_Layer/BiasAdd/ReadVariableOpReadVariableOp5fully_connected_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,Fully_Connected_Layer/BiasAdd/ReadVariableOp�
Fully_Connected_Layer/BiasAddBiasAdd&Fully_Connected_Layer/MatMul:product:04Fully_Connected_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Fully_Connected_Layer/BiasAdd�
Fully_Connected_Layer/ReluRelu&Fully_Connected_Layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
Fully_Connected_Layer/Relu�
"Output_Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"Output_Layer/MatMul/ReadVariableOp�
Output_Layer/MatMulMatMul(Fully_Connected_Layer/Relu:activations:0*Output_Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Output_Layer/MatMul�
#Output_Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Output_Layer/BiasAdd/ReadVariableOp�
Output_Layer/BiasAddBiasAddOutput_Layer/MatMul:product:0+Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Output_Layer/BiasAdd�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_layer_2_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_layer_3_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_layer_4_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentityOutput_Layer/BiasAdd:output:0$^Conv_Layer_1/BiasAdd/ReadVariableOp0^Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp$^Conv_Layer_2/BiasAdd/ReadVariableOp0^Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp6^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp$^Conv_Layer_3/BiasAdd/ReadVariableOp0^Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp6^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp$^Conv_Layer_4/BiasAdd/ReadVariableOp0^Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp6^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp-^Fully_Connected_Layer/BiasAdd/ReadVariableOp,^Fully_Connected_Layer/MatMul/ReadVariableOp$^Output_Layer/BiasAdd/ReadVariableOp#^Output_Layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::2J
#Conv_Layer_1/BiasAdd/ReadVariableOp#Conv_Layer_1/BiasAdd/ReadVariableOp2b
/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp/Conv_Layer_1/conv1d/ExpandDims_1/ReadVariableOp2J
#Conv_Layer_2/BiasAdd/ReadVariableOp#Conv_Layer_2/BiasAdd/ReadVariableOp2b
/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp/Conv_Layer_2/conv1d/ExpandDims_1/ReadVariableOp2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp2J
#Conv_Layer_3/BiasAdd/ReadVariableOp#Conv_Layer_3/BiasAdd/ReadVariableOp2b
/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp/Conv_Layer_3/conv1d/ExpandDims_1/ReadVariableOp2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp2J
#Conv_Layer_4/BiasAdd/ReadVariableOp#Conv_Layer_4/BiasAdd/ReadVariableOp2b
/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp/Conv_Layer_4/conv1d/ExpandDims_1/ReadVariableOp2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp2\
,Fully_Connected_Layer/BiasAdd/ReadVariableOp,Fully_Connected_Layer/BiasAdd/ReadVariableOp2Z
+Fully_Connected_Layer/MatMul/ReadVariableOp+Fully_Connected_Layer/MatMul/ReadVariableOp2J
#Output_Layer/BiasAdd/ReadVariableOp#Output_Layer/BiasAdd/ReadVariableOp2H
"Output_Layer/MatMul/ReadVariableOp"Output_Layer/MatMul/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
l
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_12818913

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
l
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_12818981

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:����������2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:����������2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�N
�
A__inference_CNN_layer_call_and_return_conditional_losses_12819320

inputs
conv_layer_1_12819268
conv_layer_1_12819270
conv_layer_2_12819273
conv_layer_2_12819275
conv_layer_3_12819279
conv_layer_3_12819281
conv_layer_4_12819285
conv_layer_4_12819287"
fully_connected_layer_12819291"
fully_connected_layer_12819293
output_layer_12819296
output_layer_12819298
identity��$Conv_Layer_1/StatefulPartitionedCall�$Conv_Layer_2/StatefulPartitionedCall�5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�$Conv_Layer_3/StatefulPartitionedCall�5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�$Conv_Layer_4/StatefulPartitionedCall�5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�-Fully_Connected_Layer/StatefulPartitionedCall�$Output_Layer/StatefulPartitionedCall�
$Conv_Layer_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_layer_1_12819268conv_layer_1_12819270*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_128188472&
$Conv_Layer_1/StatefulPartitionedCall�
$Conv_Layer_2/StatefulPartitionedCallStatefulPartitionedCall-Conv_Layer_1/StatefulPartitionedCall:output:0conv_layer_2_12819273conv_layer_2_12819275*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_128188852&
$Conv_Layer_2/StatefulPartitionedCall�
Dropout_Layer_1/PartitionedCallPartitionedCall-Conv_Layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_128189182!
Dropout_Layer_1/PartitionedCall�
$Conv_Layer_3/StatefulPartitionedCallStatefulPartitionedCall(Dropout_Layer_1/PartitionedCall:output:0conv_layer_3_12819279conv_layer_3_12819281*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_128189532&
$Conv_Layer_3/StatefulPartitionedCall�
Dropout_Layer_2/PartitionedCallPartitionedCall-Conv_Layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_128189862!
Dropout_Layer_2/PartitionedCall�
$Conv_Layer_4/StatefulPartitionedCallStatefulPartitionedCall(Dropout_Layer_2/PartitionedCall:output:0conv_layer_4_12819285conv_layer_4_12819287*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_128190212&
$Conv_Layer_4/StatefulPartitionedCall�
Flatten_Layer/PartitionedCallPartitionedCall-Conv_Layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_128190432
Flatten_Layer/PartitionedCall�
-Fully_Connected_Layer/StatefulPartitionedCallStatefulPartitionedCall&Flatten_Layer/PartitionedCall:output:0fully_connected_layer_12819291fully_connected_layer_12819293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_128190622/
-Fully_Connected_Layer/StatefulPartitionedCall�
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall6Fully_Connected_Layer/StatefulPartitionedCall:output:0output_layer_12819296output_layer_12819298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Output_Layer_layer_call_and_return_conditional_losses_128190882&
$Output_Layer/StatefulPartitionedCall�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_2_12819273*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_3_12819279*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_4_12819285*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0%^Conv_Layer_1/StatefulPartitionedCall%^Conv_Layer_2/StatefulPartitionedCall6^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp%^Conv_Layer_3/StatefulPartitionedCall6^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp%^Conv_Layer_4/StatefulPartitionedCall6^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp.^Fully_Connected_Layer/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::2L
$Conv_Layer_1/StatefulPartitionedCall$Conv_Layer_1/StatefulPartitionedCall2L
$Conv_Layer_2/StatefulPartitionedCall$Conv_Layer_2/StatefulPartitionedCall2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$Conv_Layer_3/StatefulPartitionedCall$Conv_Layer_3/StatefulPartitionedCall2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$Conv_Layer_4/StatefulPartitionedCall$Conv_Layer_4/StatefulPartitionedCall2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp2^
-Fully_Connected_Layer/StatefulPartitionedCall-Fully_Connected_Layer/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_Conv_Layer_2_layer_call_fn_12819712

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_128188852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_12819062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
2__inference_Dropout_Layer_1_layer_call_fn_12819734

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_128189132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�N
�
A__inference_CNN_layer_call_and_return_conditional_losses_12819178
input_layer
conv_layer_1_12819126
conv_layer_1_12819128
conv_layer_2_12819131
conv_layer_2_12819133
conv_layer_3_12819137
conv_layer_3_12819139
conv_layer_4_12819143
conv_layer_4_12819145"
fully_connected_layer_12819149"
fully_connected_layer_12819151
output_layer_12819154
output_layer_12819156
identity��$Conv_Layer_1/StatefulPartitionedCall�$Conv_Layer_2/StatefulPartitionedCall�5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�$Conv_Layer_3/StatefulPartitionedCall�5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�$Conv_Layer_4/StatefulPartitionedCall�5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�-Fully_Connected_Layer/StatefulPartitionedCall�$Output_Layer/StatefulPartitionedCall�
$Conv_Layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_layer_1_12819126conv_layer_1_12819128*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_128188472&
$Conv_Layer_1/StatefulPartitionedCall�
$Conv_Layer_2/StatefulPartitionedCallStatefulPartitionedCall-Conv_Layer_1/StatefulPartitionedCall:output:0conv_layer_2_12819131conv_layer_2_12819133*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_128188852&
$Conv_Layer_2/StatefulPartitionedCall�
Dropout_Layer_1/PartitionedCallPartitionedCall-Conv_Layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_128189182!
Dropout_Layer_1/PartitionedCall�
$Conv_Layer_3/StatefulPartitionedCallStatefulPartitionedCall(Dropout_Layer_1/PartitionedCall:output:0conv_layer_3_12819137conv_layer_3_12819139*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_128189532&
$Conv_Layer_3/StatefulPartitionedCall�
Dropout_Layer_2/PartitionedCallPartitionedCall-Conv_Layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_128189862!
Dropout_Layer_2/PartitionedCall�
$Conv_Layer_4/StatefulPartitionedCallStatefulPartitionedCall(Dropout_Layer_2/PartitionedCall:output:0conv_layer_4_12819143conv_layer_4_12819145*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_128190212&
$Conv_Layer_4/StatefulPartitionedCall�
Flatten_Layer/PartitionedCallPartitionedCall-Conv_Layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_128190432
Flatten_Layer/PartitionedCall�
-Fully_Connected_Layer/StatefulPartitionedCallStatefulPartitionedCall&Flatten_Layer/PartitionedCall:output:0fully_connected_layer_12819149fully_connected_layer_12819151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *\
fWRU
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_128190622/
-Fully_Connected_Layer/StatefulPartitionedCall�
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall6Fully_Connected_Layer/StatefulPartitionedCall:output:0output_layer_12819154output_layer_12819156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_Output_Layer_layer_call_and_return_conditional_losses_128190882&
$Output_Layer/StatefulPartitionedCall�
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_2_12819131*#
_output_shapes
:@�*
dtype027
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_2/kernel/Regularizer/SquareSquare=Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:@�2(
&Conv_Layer_2/kernel/Regularizer/Square�
%Conv_Layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_2/kernel/Regularizer/Const�
#Conv_Layer_2/kernel/Regularizer/SumSum*Conv_Layer_2/kernel/Regularizer/Square:y:0.Conv_Layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/Sum�
%Conv_Layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_2/kernel/Regularizer/mul/x�
#Conv_Layer_2/kernel/Regularizer/mulMul.Conv_Layer_2/kernel/Regularizer/mul/x:output:0,Conv_Layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_2/kernel/Regularizer/mul�
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_3_12819137*$
_output_shapes
:��*
dtype027
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_3/kernel/Regularizer/SquareSquare=Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_3/kernel/Regularizer/Square�
%Conv_Layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_3/kernel/Regularizer/Const�
#Conv_Layer_3/kernel/Regularizer/SumSum*Conv_Layer_3/kernel/Regularizer/Square:y:0.Conv_Layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/Sum�
%Conv_Layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_3/kernel/Regularizer/mul/x�
#Conv_Layer_3/kernel/Regularizer/mulMul.Conv_Layer_3/kernel/Regularizer/mul/x:output:0,Conv_Layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_3/kernel/Regularizer/mul�
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_layer_4_12819143*$
_output_shapes
:��*
dtype027
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp�
&Conv_Layer_4/kernel/Regularizer/SquareSquare=Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*$
_output_shapes
:��2(
&Conv_Layer_4/kernel/Regularizer/Square�
%Conv_Layer_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%Conv_Layer_4/kernel/Regularizer/Const�
#Conv_Layer_4/kernel/Regularizer/SumSum*Conv_Layer_4/kernel/Regularizer/Square:y:0.Conv_Layer_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/Sum�
%Conv_Layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2'
%Conv_Layer_4/kernel/Regularizer/mul/x�
#Conv_Layer_4/kernel/Regularizer/mulMul.Conv_Layer_4/kernel/Regularizer/mul/x:output:0,Conv_Layer_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#Conv_Layer_4/kernel/Regularizer/mul�
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0%^Conv_Layer_1/StatefulPartitionedCall%^Conv_Layer_2/StatefulPartitionedCall6^Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp%^Conv_Layer_3/StatefulPartitionedCall6^Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp%^Conv_Layer_4/StatefulPartitionedCall6^Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp.^Fully_Connected_Layer/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:���������::::::::::::2L
$Conv_Layer_1/StatefulPartitionedCall$Conv_Layer_1/StatefulPartitionedCall2L
$Conv_Layer_2/StatefulPartitionedCall$Conv_Layer_2/StatefulPartitionedCall2n
5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$Conv_Layer_3/StatefulPartitionedCall$Conv_Layer_3/StatefulPartitionedCall2n
5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$Conv_Layer_4/StatefulPartitionedCall$Conv_Layer_4/StatefulPartitionedCall2n
5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp5Conv_Layer_4/kernel/Regularizer/Square/ReadVariableOp2^
-Fully_Connected_Layer/StatefulPartitionedCall-Fully_Connected_Layer/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_nameInput_Layer
�	
�
J__inference_Output_Layer_layer_call_and_return_conditional_losses_12819088

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
k
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_12818918

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:����������2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
Input_Layer8
serving_default_Input_Layer:0���������@
Output_Layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�^
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
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
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�Z
_tf_keras_network�Z{"class_name": "Functional", "name": "CNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CNN", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 19, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_Layer"}, "name": "Input_Layer", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "Conv_Layer_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_Layer_1", "inbound_nodes": [[["Input_Layer", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv_Layer_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_Layer_2", "inbound_nodes": [[["Conv_Layer_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout_Layer_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "Dropout_Layer_1", "inbound_nodes": [[["Conv_Layer_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv_Layer_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_Layer_3", "inbound_nodes": [[["Dropout_Layer_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout_Layer_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "Dropout_Layer_2", "inbound_nodes": [[["Conv_Layer_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv_Layer_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_Layer_4", "inbound_nodes": [[["Dropout_Layer_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "Flatten_Layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Flatten_Layer", "inbound_nodes": [[["Conv_Layer_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Fully_Connected_Layer", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Fully_Connected_Layer", "inbound_nodes": [[["Flatten_Layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Output_Layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Output_Layer", "inbound_nodes": [[["Fully_Connected_Layer", 0, 0, {}]]]}], "input_layers": [["Input_Layer", 0, 0]], "output_layers": [["Output_Layer", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 19, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CNN", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 19, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_Layer"}, "name": "Input_Layer", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "Conv_Layer_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_Layer_1", "inbound_nodes": [[["Input_Layer", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv_Layer_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_Layer_2", "inbound_nodes": [[["Conv_Layer_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout_Layer_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "Dropout_Layer_1", "inbound_nodes": [[["Conv_Layer_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv_Layer_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_Layer_3", "inbound_nodes": [[["Dropout_Layer_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout_Layer_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "Dropout_Layer_2", "inbound_nodes": [[["Conv_Layer_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv_Layer_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv_Layer_4", "inbound_nodes": [[["Dropout_Layer_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "Flatten_Layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Flatten_Layer", "inbound_nodes": [[["Conv_Layer_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Fully_Connected_Layer", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Fully_Connected_Layer", "inbound_nodes": [[["Flatten_Layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Output_Layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Output_Layer", "inbound_nodes": [[["Fully_Connected_Layer", 0, 0, {}]]]}], "input_layers": [["Input_Layer", 0, 0]], "output_layers": [["Output_Layer", 0, 0]]}}, "training_config": {"loss": "mae", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 2.9999999242136255e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "Input_Layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 19, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 19, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_Layer"}}
�	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "Conv_Layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_Layer_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 1]}}
�


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv1D", "name": "Conv_Layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_Layer_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 64]}}
�
trainable_variables
	variables
regularization_losses
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "Dropout_Layer_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dropout_Layer_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�


!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv1D", "name": "Conv_Layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_Layer_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 128]}}
�
'trainable_variables
(	variables
)regularization_losses
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "Dropout_Layer_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dropout_Layer_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�


+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"class_name": "Conv1D", "name": "Conv_Layer_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv_Layer_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 128]}}
�
1trainable_variables
2	variables
3regularization_losses
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "Flatten_Layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Flatten_Layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

5kernel
6bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Fully_Connected_Layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Fully_Connected_Layer", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�

;kernel
<bias
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Output_Layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Output_Layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
�
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratem�m�m�m�!m�"m�+m�,m�5m�6m�;m�<m�v�v�v�v�!v�"v�+v�,v�5v�6v�;v�<v�"
	optimizer
v
0
1
2
3
!4
"5
+6
,7
58
69
;10
<11"
trackable_list_wrapper
v
0
1
2
3
!4
"5
+6
,7
58
69
;10
<11"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
�
Flayer_metrics
trainable_variables
Gnon_trainable_variables
Hmetrics
	variables

Ilayers
regularization_losses
Jlayer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
):'	@2Conv_Layer_1/kernel
:@2Conv_Layer_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Klayer_metrics
trainable_variables
Lnon_trainable_variables
Mmetrics
	variables

Nlayers
regularization_losses
Olayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@�2Conv_Layer_2/kernel
 :�2Conv_Layer_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Player_metrics
trainable_variables
Qnon_trainable_variables
Rmetrics
	variables

Slayers
regularization_losses
Tlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ulayer_metrics
trainable_variables
Vnon_trainable_variables
Wmetrics
	variables

Xlayers
regularization_losses
Ylayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2Conv_Layer_3/kernel
 :�2Conv_Layer_3/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
Zlayer_metrics
#trainable_variables
[non_trainable_variables
\metrics
$	variables

]layers
%regularization_losses
^layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
_layer_metrics
'trainable_variables
`non_trainable_variables
ametrics
(	variables

blayers
)regularization_losses
clayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2Conv_Layer_4/kernel
 :�2Conv_Layer_4/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
dlayer_metrics
-trainable_variables
enon_trainable_variables
fmetrics
.	variables

glayers
/regularization_losses
hlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ilayer_metrics
1trainable_variables
jnon_trainable_variables
kmetrics
2	variables

llayers
3regularization_losses
mlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
/:-	�2Fully_Connected_Layer/kernel
(:&2Fully_Connected_Layer/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
nlayer_metrics
7trainable_variables
onon_trainable_variables
pmetrics
8	variables

qlayers
9regularization_losses
rlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
%:#2Output_Layer/kernel
:2Output_Layer/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
slayer_metrics
=trainable_variables
tnon_trainable_variables
umetrics
>	variables

vlayers
?regularization_losses
wlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
x0
y1
z2"
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
(
�0"
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
(
�0"
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
(
�0"
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
�
	{total
	|count
}	variables
~	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
:  (2total
:  (2count
.
{0
|1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/
0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
.:,	@2Adam/Conv_Layer_1/kernel/m
$:"@2Adam/Conv_Layer_1/bias/m
/:-@�2Adam/Conv_Layer_2/kernel/m
%:#�2Adam/Conv_Layer_2/bias/m
0:.��2Adam/Conv_Layer_3/kernel/m
%:#�2Adam/Conv_Layer_3/bias/m
0:.��2Adam/Conv_Layer_4/kernel/m
%:#�2Adam/Conv_Layer_4/bias/m
4:2	�2#Adam/Fully_Connected_Layer/kernel/m
-:+2!Adam/Fully_Connected_Layer/bias/m
*:(2Adam/Output_Layer/kernel/m
$:"2Adam/Output_Layer/bias/m
.:,	@2Adam/Conv_Layer_1/kernel/v
$:"@2Adam/Conv_Layer_1/bias/v
/:-@�2Adam/Conv_Layer_2/kernel/v
%:#�2Adam/Conv_Layer_2/bias/v
0:.��2Adam/Conv_Layer_3/kernel/v
%:#�2Adam/Conv_Layer_3/bias/v
0:.��2Adam/Conv_Layer_4/kernel/v
%:#�2Adam/Conv_Layer_4/bias/v
4:2	�2#Adam/Fully_Connected_Layer/kernel/v
-:+2!Adam/Fully_Connected_Layer/bias/v
*:(2Adam/Output_Layer/kernel/v
$:"2Adam/Output_Layer/bias/v
�2�
A__inference_CNN_layer_call_and_return_conditional_losses_12819505
A__inference_CNN_layer_call_and_return_conditional_losses_12819123
A__inference_CNN_layer_call_and_return_conditional_losses_12819592
A__inference_CNN_layer_call_and_return_conditional_losses_12819178�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_CNN_layer_call_fn_12819347
&__inference_CNN_layer_call_fn_12819621
&__inference_CNN_layer_call_fn_12819650
&__inference_CNN_layer_call_fn_12819263�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference__wrapped_model_12818827�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
Input_Layer���������
�2�
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_12819666�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Conv_Layer_1_layer_call_fn_12819675�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_12819703�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Conv_Layer_2_layer_call_fn_12819712�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_12819724
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_12819729�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_Dropout_Layer_1_layer_call_fn_12819739
2__inference_Dropout_Layer_1_layer_call_fn_12819734�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_12819767�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Conv_Layer_3_layer_call_fn_12819776�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_12819793
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_12819788�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_Dropout_Layer_2_layer_call_fn_12819803
2__inference_Dropout_Layer_2_layer_call_fn_12819798�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_12819831�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Conv_Layer_4_layer_call_fn_12819840�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_12819846�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_Flatten_Layer_layer_call_fn_12819851�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_12819862�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_Fully_Connected_Layer_layer_call_fn_12819871�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_Output_Layer_layer_call_and_return_conditional_losses_12819881�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Output_Layer_layer_call_fn_12819890�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_12819901�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_12819912�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_12819923�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
&__inference_signature_wrapper_12819404Input_Layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
A__inference_CNN_layer_call_and_return_conditional_losses_12819123w!"+,56;<@�=
6�3
)�&
Input_Layer���������
p

 
� "%�"
�
0���������
� �
A__inference_CNN_layer_call_and_return_conditional_losses_12819178w!"+,56;<@�=
6�3
)�&
Input_Layer���������
p 

 
� "%�"
�
0���������
� �
A__inference_CNN_layer_call_and_return_conditional_losses_12819505r!"+,56;<;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
A__inference_CNN_layer_call_and_return_conditional_losses_12819592r!"+,56;<;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
&__inference_CNN_layer_call_fn_12819263j!"+,56;<@�=
6�3
)�&
Input_Layer���������
p

 
� "�����������
&__inference_CNN_layer_call_fn_12819347j!"+,56;<@�=
6�3
)�&
Input_Layer���������
p 

 
� "�����������
&__inference_CNN_layer_call_fn_12819621e!"+,56;<;�8
1�.
$�!
inputs���������
p

 
� "�����������
&__inference_CNN_layer_call_fn_12819650e!"+,56;<;�8
1�.
$�!
inputs���������
p 

 
� "�����������
J__inference_Conv_Layer_1_layer_call_and_return_conditional_losses_12819666d3�0
)�&
$�!
inputs���������
� ")�&
�
0���������@
� �
/__inference_Conv_Layer_1_layer_call_fn_12819675W3�0
)�&
$�!
inputs���������
� "����������@�
J__inference_Conv_Layer_2_layer_call_and_return_conditional_losses_12819703e3�0
)�&
$�!
inputs���������@
� "*�'
 �
0����������
� �
/__inference_Conv_Layer_2_layer_call_fn_12819712X3�0
)�&
$�!
inputs���������@
� "������������
J__inference_Conv_Layer_3_layer_call_and_return_conditional_losses_12819767f!"4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
/__inference_Conv_Layer_3_layer_call_fn_12819776Y!"4�1
*�'
%�"
inputs����������
� "������������
J__inference_Conv_Layer_4_layer_call_and_return_conditional_losses_12819831f+,4�1
*�'
%�"
inputs����������
� "*�'
 �
0����������
� �
/__inference_Conv_Layer_4_layer_call_fn_12819840Y+,4�1
*�'
%�"
inputs����������
� "������������
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_12819724f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
M__inference_Dropout_Layer_1_layer_call_and_return_conditional_losses_12819729f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
2__inference_Dropout_Layer_1_layer_call_fn_12819734Y8�5
.�+
%�"
inputs����������
p
� "������������
2__inference_Dropout_Layer_1_layer_call_fn_12819739Y8�5
.�+
%�"
inputs����������
p 
� "������������
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_12819788f8�5
.�+
%�"
inputs����������
p
� "*�'
 �
0����������
� �
M__inference_Dropout_Layer_2_layer_call_and_return_conditional_losses_12819793f8�5
.�+
%�"
inputs����������
p 
� "*�'
 �
0����������
� �
2__inference_Dropout_Layer_2_layer_call_fn_12819798Y8�5
.�+
%�"
inputs����������
p
� "������������
2__inference_Dropout_Layer_2_layer_call_fn_12819803Y8�5
.�+
%�"
inputs����������
p 
� "������������
K__inference_Flatten_Layer_layer_call_and_return_conditional_losses_12819846^4�1
*�'
%�"
inputs����������
� "&�#
�
0����������
� �
0__inference_Flatten_Layer_layer_call_fn_12819851Q4�1
*�'
%�"
inputs����������
� "������������
S__inference_Fully_Connected_Layer_layer_call_and_return_conditional_losses_12819862]560�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
8__inference_Fully_Connected_Layer_layer_call_fn_12819871P560�-
&�#
!�
inputs����������
� "�����������
J__inference_Output_Layer_layer_call_and_return_conditional_losses_12819881\;</�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
/__inference_Output_Layer_layer_call_fn_12819890O;</�,
%�"
 �
inputs���������
� "�����������
#__inference__wrapped_model_12818827�!"+,56;<8�5
.�+
)�&
Input_Layer���������
� ";�8
6
Output_Layer&�#
Output_Layer���������=
__inference_loss_fn_0_12819901�

� 
� "� =
__inference_loss_fn_1_12819912!�

� 
� "� =
__inference_loss_fn_2_12819923+�

� 
� "� �
&__inference_signature_wrapper_12819404�!"+,56;<G�D
� 
=�:
8
Input_Layer)�&
Input_Layer���������";�8
6
Output_Layer&�#
Output_Layer���������