��1
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
�
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
.
Log1p
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
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
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	
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
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
.
Rsqrt
x"T
y"T"
Ttype:

2
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8��+
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
h

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
h

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
h

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
h

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_4
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
h

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_5
a
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
: *
dtype0
h

Variable_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_6
a
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
: *
dtype0
h

Variable_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_7
a
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
: *
dtype0
h

Variable_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_8
a
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
: *
dtype0
h

Variable_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_9
a
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
: *
dtype0
j
Variable_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_10
c
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
: *
dtype0
j
Variable_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_11
c
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
: *
dtype0
j
Variable_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_12
c
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
: *
dtype0
j
Variable_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_13
c
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes
: *
dtype0
j
Variable_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_14
c
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes
: *
dtype0
j
Variable_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_15
c
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes
: *
dtype0
j
Variable_16VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_16
c
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes
: *
dtype0
j
Variable_17VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_17
c
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes
: *
dtype0
j
Variable_18VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_18
c
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes
: *
dtype0
j
Variable_19VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_19
c
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*
_output_shapes
: *
dtype0
j
Variable_20VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameVariable_20
c
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
s
p_re_lu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namep_re_lu/alpha
l
!p_re_lu/alpha/Read/ReadVariableOpReadVariableOpp_re_lu/alpha*
_output_shapes	
:�*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�u
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�t
value�tB�t B�t
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer_with_weights-5
layer-27
layer_with_weights-6
layer-28
layer_with_weights-7
layer-29
layer-30
 layer_with_weights-8
 layer-31
!	optimizer
"trainable_variables
#regularization_losses
$	variables
%	keras_api
&
signatures
 
�

'mu_t_1

(mu_t_2

)mu_t_3

*mu_t_4
+	sigma_t_1
,	sigma_t_2
-	sigma_t_3
.	sigma_t_4
/	sigma_t_5
0	sigma_t_6
1	sigma_t_7
2	sigma_t_8

3mu_p_1

4mu_p_2

5mu_p_3
6	sigma_p_1
7	sigma_p_2
8	sigma_p_3
9	sigma_p_4
:	sigma_p_5
;	sigma_p_6
<trainable_variables
=regularization_losses
>	variables
?	keras_api
h

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
h

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
�
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
�
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api

^	keras_api

_	keras_api

`	keras_api

a	keras_api
R
btrainable_variables
cregularization_losses
d	variables
e	keras_api
R
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
R
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
R
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
R
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
R
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
R
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
T
~trainable_variables
regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
b

�alpha
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
 
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
@21
A22
F23
G24
M25
N26
V27
W28
�29
�30
�31
�32
�33
�34
�35
 
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
@21
A22
F23
G24
M25
N26
O27
P28
V29
W30
X31
Y32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�
"trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
#regularization_losses
�metrics
�layer_metrics
$	variables
 
TR
VARIABLE_VALUEVariable6layer_with_weights-0/mu_t_1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
Variable_16layer_with_weights-0/mu_t_2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
Variable_26layer_with_weights-0/mu_t_3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUE
Variable_36layer_with_weights-0/mu_t_4/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE
Variable_49layer_with_weights-0/sigma_t_1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE
Variable_59layer_with_weights-0/sigma_t_2/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE
Variable_69layer_with_weights-0/sigma_t_3/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE
Variable_79layer_with_weights-0/sigma_t_4/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE
Variable_89layer_with_weights-0/sigma_t_5/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE
Variable_99layer_with_weights-0/sigma_t_6/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEVariable_109layer_with_weights-0/sigma_t_7/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEVariable_119layer_with_weights-0/sigma_t_8/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEVariable_126layer_with_weights-0/mu_p_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEVariable_136layer_with_weights-0/mu_p_2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEVariable_146layer_with_weights-0/mu_p_3/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEVariable_159layer_with_weights-0/sigma_p_1/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEVariable_169layer_with_weights-0/sigma_p_2/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEVariable_179layer_with_weights-0/sigma_p_3/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEVariable_189layer_with_weights-0/sigma_p_4/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEVariable_199layer_with_weights-0/sigma_p_5/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEVariable_209layer_with_weights-0/sigma_p_6/.ATTRIBUTES/VARIABLE_VALUE
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
 
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
�
<trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
=regularization_losses
�metrics
�layer_metrics
>	variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
�
Btrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
Cregularization_losses
�metrics
�layer_metrics
D	variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1
 

F0
G1
�
Htrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
Iregularization_losses
�metrics
�layer_metrics
J	variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
 

M0
N1
O2
P3
�
Qtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
Rregularization_losses
�metrics
�layer_metrics
S	variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
X2
Y3
�
Ztrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
[regularization_losses
�metrics
�layer_metrics
\	variables
 
 
 
 
 
 
 
�
btrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
cregularization_losses
�metrics
�layer_metrics
d	variables
 
 
 
�
ftrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
gregularization_losses
�metrics
�layer_metrics
h	variables
 
 
 
�
jtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
kregularization_losses
�metrics
�layer_metrics
l	variables
 
 
 
�
ntrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
oregularization_losses
�metrics
�layer_metrics
p	variables
 
 
 
�
rtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
sregularization_losses
�metrics
�layer_metrics
t	variables
 
 
 
�
vtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
wregularization_losses
�metrics
�layer_metrics
x	variables
 
 
 
�
ztrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
{regularization_losses
�metrics
�layer_metrics
|	variables
 
 
 
�
~trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
XV
VARIABLE_VALUEp_re_lu/alpha5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUE

�0
 

�0
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
 
 
 
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
,
O0
P1
X2
Y3
�4
�5
 
�
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
22
23
24
25
26
27
28
29
30
 31
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
O0
P1
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

�0
�1
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
�
serving_default_input_1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable
Variable_4Variable_12Variable_15
Variable_1
Variable_5Variable_13Variable_16
Variable_2
Variable_6Variable_14Variable_17
Variable_3
Variable_7
Variable_8Variable_18
Variable_9Variable_19Variable_10Variable_20Variable_11conv2d_1/kernelconv2d_1/biasconv2d/kernelconv2d/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betap_re_lu/alphadense_1/kerneldense_1/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_3321
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_4/Read/ReadVariableOpVariable_5/Read/ReadVariableOpVariable_6/Read/ReadVariableOpVariable_7/Read/ReadVariableOpVariable_8/Read/ReadVariableOpVariable_9/Read/ReadVariableOpVariable_10/Read/ReadVariableOpVariable_11/Read/ReadVariableOpVariable_12/Read/ReadVariableOpVariable_13/Read/ReadVariableOpVariable_14/Read/ReadVariableOpVariable_15/Read/ReadVariableOpVariable_16/Read/ReadVariableOpVariable_17/Read/ReadVariableOpVariable_18/Read/ReadVariableOpVariable_19/Read/ReadVariableOpVariable_20/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp!p_re_lu/alpha/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*7
Tin0
.2,*
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
GPU 2J 8� *&
f!R
__inference__traced_save_5572
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7
Variable_8
Variable_9Variable_10Variable_11Variable_12Variable_13Variable_14Variable_15Variable_16Variable_17Variable_18Variable_19Variable_20conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense/kernel
dense/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancep_re_lu/alphadense_1/kerneldense_1/bias*6
Tin/
-2+*
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
GPU 2J 8� *)
f$R"
 __inference__traced_restore_5708��)
�,
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4981

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_2219

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
L
0__inference_average_pooling2d_layer_call_fn_1325

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_average_pooling2d_layer_call_and_return_conditional_losses_13192
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_5054

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_5311

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_2261

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
K__inference_custom_activation_layer_call_and_return_conditional_losses_4574

inputs
sub_readvariableop_resource#
truediv_readvariableop_resource!
sub_1_readvariableop_resource%
!truediv_1_readvariableop_resource!
sub_3_readvariableop_resource%
!truediv_3_readvariableop_resource!
sub_4_readvariableop_resource%
!truediv_4_readvariableop_resource!
sub_6_readvariableop_resource%
!truediv_6_readvariableop_resource!
sub_7_readvariableop_resource%
!truediv_7_readvariableop_resource!
sub_9_readvariableop_resource%
!truediv_9_readvariableop_resource&
"truediv_10_readvariableop_resource&
"truediv_11_readvariableop_resource&
"truediv_13_readvariableop_resource&
"truediv_14_readvariableop_resource&
"truediv_16_readvariableop_resource&
"truediv_17_readvariableop_resource&
"truediv_19_readvariableop_resource
identity

identity_1��sub/ReadVariableOp�sub_1/ReadVariableOp�sub_10/ReadVariableOp�sub_11/ReadVariableOp�sub_12/ReadVariableOp�sub_13/ReadVariableOp�sub_14/ReadVariableOp�sub_15/ReadVariableOp�sub_16/ReadVariableOp�sub_17/ReadVariableOp�sub_18/ReadVariableOp�sub_19/ReadVariableOp�sub_2/ReadVariableOp�sub_3/ReadVariableOp�sub_4/ReadVariableOp�sub_5/ReadVariableOp�sub_6/ReadVariableOp�sub_7/ReadVariableOp�sub_8/ReadVariableOp�sub_9/ReadVariableOp�truediv/ReadVariableOp�truediv_1/ReadVariableOp�truediv_10/ReadVariableOp�truediv_11/ReadVariableOp�truediv_12/ReadVariableOp�truediv_13/ReadVariableOp�truediv_14/ReadVariableOp�truediv_15/ReadVariableOp�truediv_16/ReadVariableOp�truediv_17/ReadVariableOp�truediv_18/ReadVariableOp�truediv_19/ReadVariableOp�truediv_2/ReadVariableOp�truediv_3/ReadVariableOp�truediv_4/ReadVariableOp�truediv_5/ReadVariableOp�truediv_6/ReadVariableOp�truediv_7/ReadVariableOp�truediv_8/ReadVariableOp�truediv_9/ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice|
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
: *
dtype02
sub/ReadVariableOpw
subSubstrided_slice:output:0sub/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub�
truediv/ReadVariableOpReadVariableOptruediv_readvariableop_resource*
_output_shapes
: *
dtype02
truediv/ReadVariableOpx
truedivRealDivsub:z:0truediv/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
truedivS
TanhTanhtruediv:z:0*
T0*'
_output_shapes
:���������2
Tanh�
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1�
sub_1/ReadVariableOpReadVariableOpsub_1_readvariableop_resource*
_output_shapes
: *
dtype02
sub_1/ReadVariableOp
sub_1Substrided_slice_1:output:0sub_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_1�
truediv_1/ReadVariableOpReadVariableOp!truediv_1_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_1/ReadVariableOp�
	truediv_1RealDiv	sub_1:z:0 truediv_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_1Y
Tanh_1Tanhtruediv_1:z:0*
T0*'
_output_shapes
:���������2
Tanh_1�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2�
sub_2/ReadVariableOpReadVariableOpsub_1_readvariableop_resource*
_output_shapes
: *
dtype02
sub_2/ReadVariableOp
sub_2Substrided_slice_2:output:0sub_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_2�
truediv_2/ReadVariableOpReadVariableOp!truediv_1_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_2/ReadVariableOp�
	truediv_2RealDiv	sub_2:z:0 truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_2Y
Tanh_2Tanhtruediv_2:z:0*
T0*'
_output_shapes
:���������2
Tanh_2�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3�
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
: *
dtype02
sub_3/ReadVariableOp
sub_3Substrided_slice_3:output:0sub_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_3�
truediv_3/ReadVariableOpReadVariableOp!truediv_3_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_3/ReadVariableOp�
	truediv_3RealDiv	sub_3:z:0 truediv_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_3Y
Tanh_3Tanhtruediv_3:z:0*
T0*'
_output_shapes
:���������2
Tanh_3�
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_4/stack_2�
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
sub_4/ReadVariableOpReadVariableOpsub_4_readvariableop_resource*
_output_shapes
: *
dtype02
sub_4/ReadVariableOp
sub_4Substrided_slice_4:output:0sub_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_4�
truediv_4/ReadVariableOpReadVariableOp!truediv_4_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_4/ReadVariableOp�
	truediv_4RealDiv	sub_4:z:0 truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_4Y
Tanh_4Tanhtruediv_4:z:0*
T0*'
_output_shapes
:���������2
Tanh_4�
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_5/stack�
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_5/stack_1�
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_5/stack_2�
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5�
sub_5/ReadVariableOpReadVariableOpsub_4_readvariableop_resource*
_output_shapes
: *
dtype02
sub_5/ReadVariableOp
sub_5Substrided_slice_5:output:0sub_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_5�
truediv_5/ReadVariableOpReadVariableOp!truediv_4_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_5/ReadVariableOp�
	truediv_5RealDiv	sub_5:z:0 truediv_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_5Y
Tanh_5Tanhtruediv_5:z:0*
T0*'
_output_shapes
:���������2
Tanh_5�
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_6/stack�
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_6/stack_1�
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_6/stack_2�
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6�
sub_6/ReadVariableOpReadVariableOpsub_6_readvariableop_resource*
_output_shapes
: *
dtype02
sub_6/ReadVariableOp
sub_6Substrided_slice_6:output:0sub_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_6�
truediv_6/ReadVariableOpReadVariableOp!truediv_6_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_6/ReadVariableOp�
	truediv_6RealDiv	sub_6:z:0 truediv_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_6Y
Tanh_6Tanhtruediv_6:z:0*
T0*'
_output_shapes
:���������2
Tanh_6�
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_7/stack�
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_7/stack_1�
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_7/stack_2�
strided_slice_7StridedSliceinputsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7�
sub_7/ReadVariableOpReadVariableOpsub_7_readvariableop_resource*
_output_shapes
: *
dtype02
sub_7/ReadVariableOp
sub_7Substrided_slice_7:output:0sub_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_7�
truediv_7/ReadVariableOpReadVariableOp!truediv_7_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_7/ReadVariableOp�
	truediv_7RealDiv	sub_7:z:0 truediv_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_7Y
Tanh_7Tanhtruediv_7:z:0*
T0*'
_output_shapes
:���������2
Tanh_7�
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_8/stack�
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_8/stack_1�
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_8/stack_2�
strided_slice_8StridedSliceinputsstrided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8�
sub_8/ReadVariableOpReadVariableOpsub_7_readvariableop_resource*
_output_shapes
: *
dtype02
sub_8/ReadVariableOp
sub_8Substrided_slice_8:output:0sub_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_8�
truediv_8/ReadVariableOpReadVariableOp!truediv_7_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_8/ReadVariableOp�
	truediv_8RealDiv	sub_8:z:0 truediv_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_8Y
Tanh_8Tanhtruediv_8:z:0*
T0*'
_output_shapes
:���������2
Tanh_8�
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_9/stack�
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_9/stack_1�
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_9/stack_2�
strided_slice_9StridedSliceinputsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9�
sub_9/ReadVariableOpReadVariableOpsub_9_readvariableop_resource*
_output_shapes
: *
dtype02
sub_9/ReadVariableOp
sub_9Substrided_slice_9:output:0sub_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_9�
truediv_9/ReadVariableOpReadVariableOp!truediv_9_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_9/ReadVariableOp�
	truediv_9RealDiv	sub_9:z:0 truediv_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_9Y
Tanh_9Tanhtruediv_9:z:0*
T0*'
_output_shapes
:���������2
Tanh_9�
stackPackTanh:y:0
Tanh_1:y:0
Tanh_2:y:0
Tanh_3:y:0
Tanh_4:y:0
Tanh_5:y:0
Tanh_6:y:0
Tanh_7:y:0
Tanh_8:y:0
Tanh_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2
stack�
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_10/stack�
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_10/stack_1�
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_10/stack_2�
strided_slice_10StridedSlicestack:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2
Reshape/shape�
ReshapeReshapestrided_slice_10:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:���������
2	
Reshape�
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_11/stack�
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_11/stack_1�
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_11/stack_2�
strided_slice_11StridedSlicestack:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11w
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2
Reshape_1/shape�
	Reshape_1Reshapestrided_slice_11:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:���������
2
	Reshape_1e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2Reshape_1:output:0stack:output:0Reshape:output:0concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
2
concat{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2
Reshape_2/shape�
	Reshape_2Reshapeconcat:output:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������
2
	Reshape_2�
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_12/stack�
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_12/stack_1�
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_12/stack_2�
strided_slice_12StridedSliceinputsstrided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12�
sub_10/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
: *
dtype02
sub_10/ReadVariableOp�
sub_10Substrided_slice_12:output:0sub_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_10�
truediv_10/ReadVariableOpReadVariableOp"truediv_10_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_10/ReadVariableOp�

truediv_10RealDiv
sub_10:z:0!truediv_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_10V
ReluRelutruediv_10:z:0*
T0*'
_output_shapes
:���������2
Relu]
Log1pLog1pRelu:activations:0*
T0*'
_output_shapes
:���������2
Log1p�
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_13/stack�
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_13/stack_1�
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_13/stack_2�
strided_slice_13StridedSliceinputsstrided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13�
sub_11/ReadVariableOpReadVariableOpsub_1_readvariableop_resource*
_output_shapes
: *
dtype02
sub_11/ReadVariableOp�
sub_11Substrided_slice_13:output:0sub_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_11�
truediv_11/ReadVariableOpReadVariableOp"truediv_11_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_11/ReadVariableOp�

truediv_11RealDiv
sub_11:z:0!truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_11Z
Relu_1Relutruediv_11:z:0*
T0*'
_output_shapes
:���������2
Relu_1c
Log1p_1Log1pRelu_1:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_1�
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_14/stack�
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_14/stack_1�
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_14/stack_2�
strided_slice_14StridedSliceinputsstrided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14�
sub_12/ReadVariableOpReadVariableOpsub_1_readvariableop_resource*
_output_shapes
: *
dtype02
sub_12/ReadVariableOp�
sub_12Substrided_slice_14:output:0sub_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_12�
truediv_12/ReadVariableOpReadVariableOp"truediv_11_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_12/ReadVariableOp�

truediv_12RealDiv
sub_12:z:0!truediv_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_12Z
Relu_2Relutruediv_12:z:0*
T0*'
_output_shapes
:���������2
Relu_2c
Log1p_2Log1pRelu_2:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_2�
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_15/stack�
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_15/stack_1�
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_15/stack_2�
strided_slice_15StridedSliceinputsstrided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15�
sub_13/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
: *
dtype02
sub_13/ReadVariableOp�
sub_13Substrided_slice_15:output:0sub_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_13�
truediv_13/ReadVariableOpReadVariableOp"truediv_13_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_13/ReadVariableOp�

truediv_13RealDiv
sub_13:z:0!truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_13Z
Relu_3Relutruediv_13:z:0*
T0*'
_output_shapes
:���������2
Relu_3c
Log1p_3Log1pRelu_3:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_3�
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_16/stack�
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_16/stack_1�
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_16/stack_2�
strided_slice_16StridedSliceinputsstrided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16�
sub_14/ReadVariableOpReadVariableOpsub_4_readvariableop_resource*
_output_shapes
: *
dtype02
sub_14/ReadVariableOp�
sub_14Substrided_slice_16:output:0sub_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_14�
truediv_14/ReadVariableOpReadVariableOp"truediv_14_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_14/ReadVariableOp�

truediv_14RealDiv
sub_14:z:0!truediv_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_14Z
Relu_4Relutruediv_14:z:0*
T0*'
_output_shapes
:���������2
Relu_4c
Log1p_4Log1pRelu_4:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_4�
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_17/stack�
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_17/stack_1�
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_17/stack_2�
strided_slice_17StridedSliceinputsstrided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17�
sub_15/ReadVariableOpReadVariableOpsub_4_readvariableop_resource*
_output_shapes
: *
dtype02
sub_15/ReadVariableOp�
sub_15Substrided_slice_17:output:0sub_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_15�
truediv_15/ReadVariableOpReadVariableOp"truediv_14_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_15/ReadVariableOp�

truediv_15RealDiv
sub_15:z:0!truediv_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_15Z
Relu_5Relutruediv_15:z:0*
T0*'
_output_shapes
:���������2
Relu_5c
Log1p_5Log1pRelu_5:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_5�
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_18/stack�
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_18/stack_1�
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_18/stack_2�
strided_slice_18StridedSliceinputsstrided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18�
sub_16/ReadVariableOpReadVariableOpsub_6_readvariableop_resource*
_output_shapes
: *
dtype02
sub_16/ReadVariableOp�
sub_16Substrided_slice_18:output:0sub_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_16�
truediv_16/ReadVariableOpReadVariableOp"truediv_16_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_16/ReadVariableOp�

truediv_16RealDiv
sub_16:z:0!truediv_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_16Z
Relu_6Relutruediv_16:z:0*
T0*'
_output_shapes
:���������2
Relu_6c
Log1p_6Log1pRelu_6:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_6�
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_19/stack�
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_19/stack_1�
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_19/stack_2�
strided_slice_19StridedSliceinputsstrided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19�
sub_17/ReadVariableOpReadVariableOpsub_7_readvariableop_resource*
_output_shapes
: *
dtype02
sub_17/ReadVariableOp�
sub_17Substrided_slice_19:output:0sub_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_17�
truediv_17/ReadVariableOpReadVariableOp"truediv_17_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_17/ReadVariableOp�

truediv_17RealDiv
sub_17:z:0!truediv_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_17Z
Relu_7Relutruediv_17:z:0*
T0*'
_output_shapes
:���������2
Relu_7c
Log1p_7Log1pRelu_7:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_7�
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_20/stack�
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_20/stack_1�
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_20/stack_2�
strided_slice_20StridedSliceinputsstrided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20�
sub_18/ReadVariableOpReadVariableOpsub_7_readvariableop_resource*
_output_shapes
: *
dtype02
sub_18/ReadVariableOp�
sub_18Substrided_slice_20:output:0sub_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_18�
truediv_18/ReadVariableOpReadVariableOp"truediv_17_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_18/ReadVariableOp�

truediv_18RealDiv
sub_18:z:0!truediv_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_18Z
Relu_8Relutruediv_18:z:0*
T0*'
_output_shapes
:���������2
Relu_8c
Log1p_8Log1pRelu_8:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_8�
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_21/stack�
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_21/stack_1�
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_21/stack_2�
strided_slice_21StridedSliceinputsstrided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21�
sub_19/ReadVariableOpReadVariableOpsub_9_readvariableop_resource*
_output_shapes
: *
dtype02
sub_19/ReadVariableOp�
sub_19Substrided_slice_21:output:0sub_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_19�
truediv_19/ReadVariableOpReadVariableOp"truediv_19_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_19/ReadVariableOp�

truediv_19RealDiv
sub_19:z:0!truediv_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_19Z
Relu_9Relutruediv_19:z:0*
T0*'
_output_shapes
:���������2
Relu_9c
Log1p_9Log1pRelu_9:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_9�
stack_1Pack	Log1p:y:0Log1p_1:y:0Log1p_2:y:0Log1p_3:y:0Log1p_4:y:0Log1p_5:y:0Log1p_6:y:0Log1p_7:y:0Log1p_8:y:0Log1p_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2	
stack_1�
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_22/stack�
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_22/stack_1�
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_22/stack_2�
strided_slice_22StridedSlicestack_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22w
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2
Reshape_3/shape�
	Reshape_3Reshapestrided_slice_22:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:���������
2
	Reshape_3�
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_23/stack�
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_23/stack_1�
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_23/stack_2�
strided_slice_23StridedSlicestack_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23w
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2
Reshape_4/shape�
	Reshape_4Reshapestrided_slice_23:output:0Reshape_4/shape:output:0*
T0*+
_output_shapes
:���������
2
	Reshape_4i
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat_1/axis�
concat_1ConcatV2Reshape_4:output:0stack_1:output:0Reshape_3:output:0concat_1/axis:output:0*
N*
T0*+
_output_shapes
:���������
2

concat_1{
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2
Reshape_5/shape�
	Reshape_5Reshapeconcat_1:output:0Reshape_5/shape:output:0*
T0*/
_output_shapes
:���������
2
	Reshape_5�
IdentityIdentityReshape_2:output:0^sub/ReadVariableOp^sub_1/ReadVariableOp^sub_10/ReadVariableOp^sub_11/ReadVariableOp^sub_12/ReadVariableOp^sub_13/ReadVariableOp^sub_14/ReadVariableOp^sub_15/ReadVariableOp^sub_16/ReadVariableOp^sub_17/ReadVariableOp^sub_18/ReadVariableOp^sub_19/ReadVariableOp^sub_2/ReadVariableOp^sub_3/ReadVariableOp^sub_4/ReadVariableOp^sub_5/ReadVariableOp^sub_6/ReadVariableOp^sub_7/ReadVariableOp^sub_8/ReadVariableOp^sub_9/ReadVariableOp^truediv/ReadVariableOp^truediv_1/ReadVariableOp^truediv_10/ReadVariableOp^truediv_11/ReadVariableOp^truediv_12/ReadVariableOp^truediv_13/ReadVariableOp^truediv_14/ReadVariableOp^truediv_15/ReadVariableOp^truediv_16/ReadVariableOp^truediv_17/ReadVariableOp^truediv_18/ReadVariableOp^truediv_19/ReadVariableOp^truediv_2/ReadVariableOp^truediv_3/ReadVariableOp^truediv_4/ReadVariableOp^truediv_5/ReadVariableOp^truediv_6/ReadVariableOp^truediv_7/ReadVariableOp^truediv_8/ReadVariableOp^truediv_9/ReadVariableOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1IdentityReshape_5:output:0^sub/ReadVariableOp^sub_1/ReadVariableOp^sub_10/ReadVariableOp^sub_11/ReadVariableOp^sub_12/ReadVariableOp^sub_13/ReadVariableOp^sub_14/ReadVariableOp^sub_15/ReadVariableOp^sub_16/ReadVariableOp^sub_17/ReadVariableOp^sub_18/ReadVariableOp^sub_19/ReadVariableOp^sub_2/ReadVariableOp^sub_3/ReadVariableOp^sub_4/ReadVariableOp^sub_5/ReadVariableOp^sub_6/ReadVariableOp^sub_7/ReadVariableOp^sub_8/ReadVariableOp^sub_9/ReadVariableOp^truediv/ReadVariableOp^truediv_1/ReadVariableOp^truediv_10/ReadVariableOp^truediv_11/ReadVariableOp^truediv_12/ReadVariableOp^truediv_13/ReadVariableOp^truediv_14/ReadVariableOp^truediv_15/ReadVariableOp^truediv_16/ReadVariableOp^truediv_17/ReadVariableOp^truediv_18/ReadVariableOp^truediv_19/ReadVariableOp^truediv_2/ReadVariableOp^truediv_3/ReadVariableOp^truediv_4/ReadVariableOp^truediv_5/ReadVariableOp^truediv_6/ReadVariableOp^truediv_7/ReadVariableOp^truediv_8/ReadVariableOp^truediv_9/ReadVariableOp*
T0*/
_output_shapes
:���������
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:���������:::::::::::::::::::::2(
sub/ReadVariableOpsub/ReadVariableOp2,
sub_1/ReadVariableOpsub_1/ReadVariableOp2.
sub_10/ReadVariableOpsub_10/ReadVariableOp2.
sub_11/ReadVariableOpsub_11/ReadVariableOp2.
sub_12/ReadVariableOpsub_12/ReadVariableOp2.
sub_13/ReadVariableOpsub_13/ReadVariableOp2.
sub_14/ReadVariableOpsub_14/ReadVariableOp2.
sub_15/ReadVariableOpsub_15/ReadVariableOp2.
sub_16/ReadVariableOpsub_16/ReadVariableOp2.
sub_17/ReadVariableOpsub_17/ReadVariableOp2.
sub_18/ReadVariableOpsub_18/ReadVariableOp2.
sub_19/ReadVariableOpsub_19/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp2,
sub_4/ReadVariableOpsub_4/ReadVariableOp2,
sub_5/ReadVariableOpsub_5/ReadVariableOp2,
sub_6/ReadVariableOpsub_6/ReadVariableOp2,
sub_7/ReadVariableOpsub_7/ReadVariableOp2,
sub_8/ReadVariableOpsub_8/ReadVariableOp2,
sub_9/ReadVariableOpsub_9/ReadVariableOp20
truediv/ReadVariableOptruediv/ReadVariableOp24
truediv_1/ReadVariableOptruediv_1/ReadVariableOp26
truediv_10/ReadVariableOptruediv_10/ReadVariableOp26
truediv_11/ReadVariableOptruediv_11/ReadVariableOp26
truediv_12/ReadVariableOptruediv_12/ReadVariableOp26
truediv_13/ReadVariableOptruediv_13/ReadVariableOp26
truediv_14/ReadVariableOptruediv_14/ReadVariableOp26
truediv_15/ReadVariableOptruediv_15/ReadVariableOp26
truediv_16/ReadVariableOptruediv_16/ReadVariableOp26
truediv_17/ReadVariableOptruediv_17/ReadVariableOp26
truediv_18/ReadVariableOptruediv_18/ReadVariableOp26
truediv_19/ReadVariableOptruediv_19/ReadVariableOp24
truediv_2/ReadVariableOptruediv_2/ReadVariableOp24
truediv_3/ReadVariableOptruediv_3/ReadVariableOp24
truediv_4/ReadVariableOptruediv_4/ReadVariableOp24
truediv_5/ReadVariableOptruediv_5/ReadVariableOp24
truediv_6/ReadVariableOptruediv_6/ReadVariableOp24
truediv_7/ReadVariableOptruediv_7/ReadVariableOp24
truediv_8/ReadVariableOptruediv_8/ReadVariableOp24
truediv_9/ReadVariableOptruediv_9/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_4235

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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
"#&'()**-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_28502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_flatten_1_layer_call_fn_5059

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
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_22332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_batch_normalization_layer_call_fn_4761

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_11382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_1_layer_call_fn_4936

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
�
?__inference_model_layer_call_and_return_conditional_losses_2850

inputs
custom_activation_2691
custom_activation_2693
custom_activation_2695
custom_activation_2697
custom_activation_2699
custom_activation_2701
custom_activation_2703
custom_activation_2705
custom_activation_2707
custom_activation_2709
custom_activation_2711
custom_activation_2713
custom_activation_2715
custom_activation_2717
custom_activation_2719
custom_activation_2721
custom_activation_2723
custom_activation_2725
custom_activation_2727
custom_activation_2729
custom_activation_2731
conv2d_1_2735
conv2d_1_2737
conv2d_2740
conv2d_2742
batch_normalization_1_2745
batch_normalization_1_2747
batch_normalization_1_2749
batch_normalization_1_2751
batch_normalization_2754
batch_normalization_2756
batch_normalization_2758
batch_normalization_2760

dense_2784

dense_2786
batch_normalization_2_2789
batch_normalization_2_2791
batch_normalization_2_2793
batch_normalization_2_2795
p_re_lu_2798
dense_1_2802
dense_1_2804
identity��+batch_normalization/StatefulPartitionedCall�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�-batch_normalization_1/StatefulPartitionedCall�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�-batch_normalization_2/StatefulPartitionedCall�<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�)custom_activation/StatefulPartitionedCall�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�p_re_lu/StatefulPartitionedCall�
)custom_activation/StatefulPartitionedCallStatefulPartitionedCallinputscustom_activation_2691custom_activation_2693custom_activation_2695custom_activation_2697custom_activation_2699custom_activation_2701custom_activation_2703custom_activation_2705custom_activation_2707custom_activation_2709custom_activation_2711custom_activation_2713custom_activation_2715custom_activation_2717custom_activation_2719custom_activation_2721custom_activation_2723custom_activation_2725custom_activation_2727custom_activation_2729custom_activation_2731*!
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:���������
:���������
*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_custom_activation_layer_call_and_return_conditional_losses_18602+
)custom_activation/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall2custom_activation/StatefulPartitionedCall:output:1conv2d_1_2735conv2d_1_2737*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_19652"
 conv2d_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCall2custom_activation/StatefulPartitionedCall:output:0conv2d_2740conv2d_2742*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_19912 
conv2d/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_2745batch_normalization_1_2747batch_normalization_1_2749batch_normalization_1_2751*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20382/
-batch_normalization_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_2754batch_normalization_2756batch_normalization_2758batch_normalization_2760*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_21352-
+batch_normalization/StatefulPartitionedCall�
tf.math.tanh_1/TanhTanh6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.math.tanh_1/Tanh�
tf.nn.relu_1/ReluRelu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.nn.relu_1/Relu�
tf.math.tanh/TanhTanh4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.math.tanh/Tanh�
tf.nn.relu/ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.nn.relu/Relu�
#average_pooling2d_3/PartitionedCallPartitionedCalltf.math.tanh_1/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13912%
#average_pooling2d_3/PartitionedCall�
max_pooling2d_3/PartitionedCallPartitionedCalltf.math.tanh_1/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_13792!
max_pooling2d_3/PartitionedCall�
#average_pooling2d_2/PartitionedCallPartitionedCalltf.nn.relu_1/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_13672%
#average_pooling2d_2/PartitionedCall�
max_pooling2d_2/PartitionedCallPartitionedCalltf.nn.relu_1/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_13552!
max_pooling2d_2/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCalltf.math.tanh/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_13432%
#average_pooling2d_1/PartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCalltf.math.tanh/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_13312!
max_pooling2d_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalltf.nn.relu/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_average_pooling2d_layer_call_and_return_conditional_losses_13192#
!average_pooling2d/PartitionedCall�
max_pooling2d/PartitionedCallPartitionedCalltf.nn.relu/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13072
max_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_22192
flatten/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_22332
flatten_1/PartitionedCall�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_22472
flatten_2/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_22612
flatten_3/PartitionedCall�
flatten_4/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_22752
flatten_4/PartitionedCall�
flatten_5/PartitionedCallPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_5_layer_call_and_return_conditional_losses_22892
flatten_5/PartitionedCall�
flatten_6/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_6_layer_call_and_return_conditional_losses_23032
flatten_6/PartitionedCall�
flatten_7/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_7_layer_call_and_return_conditional_losses_23172
flatten_7/PartitionedCall�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_23382
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_2784
dense_2786*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_23692
dense/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_2_2789batch_normalization_2_2791batch_normalization_2_2793batch_normalization_2_2795*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_15292/
-batch_normalization_2/StatefulPartitionedCall�
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0p_re_lu_2798*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_p_re_lu_layer_call_and_return_conditional_losses_15982!
p_re_lu/StatefulPartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24352!
dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_2802dense_1_2804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24642!
dense_1/StatefulPartitionedCall�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2754*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2756*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_1_2745*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_1_2747*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_2784* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2_2793*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2_2795*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp.^batch_normalization_1/StatefulPartitionedCall=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp.^batch_normalization_2/StatefulPartitionedCall=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*^custom_activation/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2V
)custom_activation/StatefulPartitionedCall)custom_activation/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_5390I
Ebatch_normalization_1_beta_regularizer_square_readvariableop_resource
identity��<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpEbatch_normalization_1_beta_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentity.batch_normalization_1/beta/Regularizer/mul:z:0=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_5043

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
@__inference_conv2d_layer_call_and_return_conditional_losses_4633

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
y
$__inference_dense_layer_call_fn_5181

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_23692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_dropout_layer_call_fn_5326

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24402
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1138

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_5076

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_6_5423I
Ebatch_normalization_2_beta_regularizer_square_readvariableop_resource
identity��<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOpEbatch_normalization_2_beta_regularizer_square_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentity.batch_normalization_2/beta/Regularizer/mul:z:0=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp
�+
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4793

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_flatten_6_layer_call_and_return_conditional_losses_5109

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_1_layer_call_fn_1337

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_13312
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�,
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5273

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentitybatchnorm/add_1:z:0=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
{
&__inference_dense_1_layer_call_fn_5346

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_2233

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4923

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
2__inference_batch_normalization_layer_call_fn_4748

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_10952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_5065

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_batch_normalization_layer_call_fn_4849

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_21652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_flatten_2_layer_call_and_return_conditional_losses_2247

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
z
%__inference_conv2d_layer_call_fn_4642

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_19912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
?__inference_model_layer_call_and_return_conditional_losses_2685
input_1
custom_activation_2526
custom_activation_2528
custom_activation_2530
custom_activation_2532
custom_activation_2534
custom_activation_2536
custom_activation_2538
custom_activation_2540
custom_activation_2542
custom_activation_2544
custom_activation_2546
custom_activation_2548
custom_activation_2550
custom_activation_2552
custom_activation_2554
custom_activation_2556
custom_activation_2558
custom_activation_2560
custom_activation_2562
custom_activation_2564
custom_activation_2566
conv2d_1_2570
conv2d_1_2572
conv2d_2575
conv2d_2577
batch_normalization_1_2580
batch_normalization_1_2582
batch_normalization_1_2584
batch_normalization_1_2586
batch_normalization_2589
batch_normalization_2591
batch_normalization_2593
batch_normalization_2595

dense_2619

dense_2621
batch_normalization_2_2624
batch_normalization_2_2626
batch_normalization_2_2628
batch_normalization_2_2630
p_re_lu_2633
dense_1_2637
dense_1_2639
identity��+batch_normalization/StatefulPartitionedCall�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�-batch_normalization_1/StatefulPartitionedCall�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�-batch_normalization_2/StatefulPartitionedCall�<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�)custom_activation/StatefulPartitionedCall�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�dense_1/StatefulPartitionedCall�p_re_lu/StatefulPartitionedCall�
)custom_activation/StatefulPartitionedCallStatefulPartitionedCallinput_1custom_activation_2526custom_activation_2528custom_activation_2530custom_activation_2532custom_activation_2534custom_activation_2536custom_activation_2538custom_activation_2540custom_activation_2542custom_activation_2544custom_activation_2546custom_activation_2548custom_activation_2550custom_activation_2552custom_activation_2554custom_activation_2556custom_activation_2558custom_activation_2560custom_activation_2562custom_activation_2564custom_activation_2566*!
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:���������
:���������
*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_custom_activation_layer_call_and_return_conditional_losses_18602+
)custom_activation/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall2custom_activation/StatefulPartitionedCall:output:1conv2d_1_2570conv2d_1_2572*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_19652"
 conv2d_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCall2custom_activation/StatefulPartitionedCall:output:0conv2d_2575conv2d_2577*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_19912 
conv2d/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_2580batch_normalization_1_2582batch_normalization_1_2584batch_normalization_1_2586*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20682/
-batch_normalization_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_2589batch_normalization_2591batch_normalization_2593batch_normalization_2595*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_21652-
+batch_normalization/StatefulPartitionedCall�
tf.math.tanh_1/TanhTanh6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.math.tanh_1/Tanh�
tf.nn.relu_1/ReluRelu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.nn.relu_1/Relu�
tf.math.tanh/TanhTanh4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.math.tanh/Tanh�
tf.nn.relu/ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.nn.relu/Relu�
#average_pooling2d_3/PartitionedCallPartitionedCalltf.math.tanh_1/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13912%
#average_pooling2d_3/PartitionedCall�
max_pooling2d_3/PartitionedCallPartitionedCalltf.math.tanh_1/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_13792!
max_pooling2d_3/PartitionedCall�
#average_pooling2d_2/PartitionedCallPartitionedCalltf.nn.relu_1/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_13672%
#average_pooling2d_2/PartitionedCall�
max_pooling2d_2/PartitionedCallPartitionedCalltf.nn.relu_1/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_13552!
max_pooling2d_2/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCalltf.math.tanh/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_13432%
#average_pooling2d_1/PartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCalltf.math.tanh/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_13312!
max_pooling2d_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalltf.nn.relu/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_average_pooling2d_layer_call_and_return_conditional_losses_13192#
!average_pooling2d/PartitionedCall�
max_pooling2d/PartitionedCallPartitionedCalltf.nn.relu/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13072
max_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_22192
flatten/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_22332
flatten_1/PartitionedCall�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_22472
flatten_2/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_22612
flatten_3/PartitionedCall�
flatten_4/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_22752
flatten_4/PartitionedCall�
flatten_5/PartitionedCallPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_5_layer_call_and_return_conditional_losses_22892
flatten_5/PartitionedCall�
flatten_6/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_6_layer_call_and_return_conditional_losses_23032
flatten_6/PartitionedCall�
flatten_7/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_7_layer_call_and_return_conditional_losses_23172
flatten_7/PartitionedCall�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_23382
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_2619
dense_2621*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_23692
dense/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_2_2624batch_normalization_2_2626batch_normalization_2_2628batch_normalization_2_2630*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_15742/
-batch_normalization_2/StatefulPartitionedCall�
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0p_re_lu_2633*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_p_re_lu_layer_call_and_return_conditional_losses_15982!
p_re_lu/StatefulPartitionedCall�
dropout/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24402
dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_2637dense_1_2639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24642!
dense_1/StatefulPartitionedCall�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2589*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2591*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_1_2580*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_1_2582*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_2619* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2_2626*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2_2630*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp.^batch_normalization_1/StatefulPartitionedCall=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp.^batch_normalization_2/StatefulPartitionedCall=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*^custom_activation/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2V
)custom_activation/StatefulPartitionedCall)custom_activation/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
E__inference_concatenate_layer_call_and_return_conditional_losses_5138
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/7
�,
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1095

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_5_layer_call_and_return_conditional_losses_2289

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
?__inference_model_layer_call_and_return_conditional_losses_3101

inputs
custom_activation_2942
custom_activation_2944
custom_activation_2946
custom_activation_2948
custom_activation_2950
custom_activation_2952
custom_activation_2954
custom_activation_2956
custom_activation_2958
custom_activation_2960
custom_activation_2962
custom_activation_2964
custom_activation_2966
custom_activation_2968
custom_activation_2970
custom_activation_2972
custom_activation_2974
custom_activation_2976
custom_activation_2978
custom_activation_2980
custom_activation_2982
conv2d_1_2986
conv2d_1_2988
conv2d_2991
conv2d_2993
batch_normalization_1_2996
batch_normalization_1_2998
batch_normalization_1_3000
batch_normalization_1_3002
batch_normalization_3005
batch_normalization_3007
batch_normalization_3009
batch_normalization_3011

dense_3035

dense_3037
batch_normalization_2_3040
batch_normalization_2_3042
batch_normalization_2_3044
batch_normalization_2_3046
p_re_lu_3049
dense_1_3053
dense_1_3055
identity��+batch_normalization/StatefulPartitionedCall�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�-batch_normalization_1/StatefulPartitionedCall�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�-batch_normalization_2/StatefulPartitionedCall�<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�)custom_activation/StatefulPartitionedCall�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�dense_1/StatefulPartitionedCall�p_re_lu/StatefulPartitionedCall�
)custom_activation/StatefulPartitionedCallStatefulPartitionedCallinputscustom_activation_2942custom_activation_2944custom_activation_2946custom_activation_2948custom_activation_2950custom_activation_2952custom_activation_2954custom_activation_2956custom_activation_2958custom_activation_2960custom_activation_2962custom_activation_2964custom_activation_2966custom_activation_2968custom_activation_2970custom_activation_2972custom_activation_2974custom_activation_2976custom_activation_2978custom_activation_2980custom_activation_2982*!
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:���������
:���������
*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_custom_activation_layer_call_and_return_conditional_losses_18602+
)custom_activation/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall2custom_activation/StatefulPartitionedCall:output:1conv2d_1_2986conv2d_1_2988*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_19652"
 conv2d_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCall2custom_activation/StatefulPartitionedCall:output:0conv2d_2991conv2d_2993*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_19912 
conv2d/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_2996batch_normalization_1_2998batch_normalization_1_3000batch_normalization_1_3002*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20682/
-batch_normalization_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3005batch_normalization_3007batch_normalization_3009batch_normalization_3011*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_21652-
+batch_normalization/StatefulPartitionedCall�
tf.math.tanh_1/TanhTanh6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.math.tanh_1/Tanh�
tf.nn.relu_1/ReluRelu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.nn.relu_1/Relu�
tf.math.tanh/TanhTanh4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.math.tanh/Tanh�
tf.nn.relu/ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.nn.relu/Relu�
#average_pooling2d_3/PartitionedCallPartitionedCalltf.math.tanh_1/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13912%
#average_pooling2d_3/PartitionedCall�
max_pooling2d_3/PartitionedCallPartitionedCalltf.math.tanh_1/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_13792!
max_pooling2d_3/PartitionedCall�
#average_pooling2d_2/PartitionedCallPartitionedCalltf.nn.relu_1/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_13672%
#average_pooling2d_2/PartitionedCall�
max_pooling2d_2/PartitionedCallPartitionedCalltf.nn.relu_1/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_13552!
max_pooling2d_2/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCalltf.math.tanh/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_13432%
#average_pooling2d_1/PartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCalltf.math.tanh/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_13312!
max_pooling2d_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalltf.nn.relu/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_average_pooling2d_layer_call_and_return_conditional_losses_13192#
!average_pooling2d/PartitionedCall�
max_pooling2d/PartitionedCallPartitionedCalltf.nn.relu/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13072
max_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_22192
flatten/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_22332
flatten_1/PartitionedCall�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_22472
flatten_2/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_22612
flatten_3/PartitionedCall�
flatten_4/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_22752
flatten_4/PartitionedCall�
flatten_5/PartitionedCallPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_5_layer_call_and_return_conditional_losses_22892
flatten_5/PartitionedCall�
flatten_6/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_6_layer_call_and_return_conditional_losses_23032
flatten_6/PartitionedCall�
flatten_7/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_7_layer_call_and_return_conditional_losses_23172
flatten_7/PartitionedCall�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_23382
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_3035
dense_3037*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_23692
dense/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_2_3040batch_normalization_2_3042batch_normalization_2_3044batch_normalization_2_3046*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_15742/
-batch_normalization_2/StatefulPartitionedCall�
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0p_re_lu_3049*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_p_re_lu_layer_call_and_return_conditional_losses_15982!
p_re_lu/StatefulPartitionedCall�
dropout/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24402
dropout/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_3053dense_1_3055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24642!
dense_1/StatefulPartitionedCall�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_3005*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_3007*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_1_2996*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_1_2998*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_3035* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2_3042*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2_3046*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp.^batch_normalization_1/StatefulPartitionedCall=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp.^batch_normalization_2/StatefulPartitionedCall=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*^custom_activation/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2V
)custom_activation/StatefulPartitionedCall)custom_activation/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_3188
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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_31012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1965

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
E__inference_concatenate_layer_call_and_return_conditional_losses_2338

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_5379J
Fbatch_normalization_1_gamma_regularizer_square_readvariableop_resource
identity��=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpFbatch_normalization_1_gamma_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
IdentityIdentity/batch_normalization_1/gamma/Regularizer/mul:z:0>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp
�
B
&__inference_flatten_layer_call_fn_5048

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
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_22192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_2_layer_call_fn_1361

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_13552
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_5368G
Cbatch_normalization_beta_regularizer_square_readvariableop_resource
identity��:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpCbatch_normalization_beta_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentity,batch_normalization/beta/Regularizer/mul:z:0;^batch_normalization/beta/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp
�,
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4705

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
i
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_1391

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_2937
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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
"#&'()**-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_28502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�,
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1574

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentitybatchnorm/add_1:z:0=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
D
(__inference_flatten_7_layer_call_fn_5125

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
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_7_layer_call_and_return_conditional_losses_23172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_flatten_4_layer_call_fn_5092

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
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_22752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_2_layer_call_fn_5286

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_15292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
C__inference_flatten_7_layer_call_and_return_conditional_losses_2317

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4735

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�+
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2135

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
@__inference_conv2d_layer_call_and_return_conditional_losses_1991

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
D
(__inference_flatten_5_layer_call_fn_5103

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
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_5_layer_call_and_return_conditional_losses_22892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
&__inference_dropout_layer_call_fn_5321

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_average_pooling2d_1_layer_call_fn_1349

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_13432
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�,
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4893

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
g
K__inference_average_pooling2d_layer_call_and_return_conditional_losses_1319

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
 __inference__traced_restore_5708
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2!
assignvariableop_3_variable_3!
assignvariableop_4_variable_4!
assignvariableop_5_variable_5!
assignvariableop_6_variable_6!
assignvariableop_7_variable_7!
assignvariableop_8_variable_8!
assignvariableop_9_variable_9#
assignvariableop_10_variable_10#
assignvariableop_11_variable_11#
assignvariableop_12_variable_12#
assignvariableop_13_variable_13#
assignvariableop_14_variable_14#
assignvariableop_15_variable_15#
assignvariableop_16_variable_16#
assignvariableop_17_variable_17#
assignvariableop_18_variable_18#
assignvariableop_19_variable_19#
assignvariableop_20_variable_20%
!assignvariableop_21_conv2d_kernel#
assignvariableop_22_conv2d_bias'
#assignvariableop_23_conv2d_1_kernel%
!assignvariableop_24_conv2d_1_bias1
-assignvariableop_25_batch_normalization_gamma0
,assignvariableop_26_batch_normalization_beta7
3assignvariableop_27_batch_normalization_moving_mean;
7assignvariableop_28_batch_normalization_moving_variance3
/assignvariableop_29_batch_normalization_1_gamma2
.assignvariableop_30_batch_normalization_1_beta9
5assignvariableop_31_batch_normalization_1_moving_mean=
9assignvariableop_32_batch_normalization_1_moving_variance$
 assignvariableop_33_dense_kernel"
assignvariableop_34_dense_bias3
/assignvariableop_35_batch_normalization_2_gamma2
.assignvariableop_36_batch_normalization_2_beta9
5assignvariableop_37_batch_normalization_2_moving_mean=
9assignvariableop_38_batch_normalization_2_moving_variance%
!assignvariableop_39_p_re_lu_alpha&
"assignvariableop_40_dense_1_kernel$
 assignvariableop_41_dense_1_bias
identity_43��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/mu_t_1/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_t_2/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_t_3/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_t_4/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_1/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_2/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_3/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_4/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_5/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_6/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_7/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_8/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_p_1/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_p_2/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_p_3/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_1/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_2/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_3/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_4/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_5/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_6/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_3Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_4Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_5Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_6Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_7Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_8Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_9Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_10Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_11Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_12Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_13Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_14Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_15Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_16Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_17Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_18Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_19Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_20Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_conv2d_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp#assignvariableop_23_conv2d_1_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv2d_1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp-assignvariableop_25_batch_normalization_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_batch_normalization_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp3assignvariableop_27_batch_normalization_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp7assignvariableop_28_batch_normalization_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_1_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp.assignvariableop_30_batch_normalization_1_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp5assignvariableop_31_batch_normalization_1_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp9assignvariableop_32_batch_normalization_1_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOpassignvariableop_34_dense_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batch_normalization_2_gammaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp.assignvariableop_36_batch_normalization_2_betaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp5assignvariableop_37_batch_normalization_2_moving_meanIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp9assignvariableop_38_batch_normalization_2_moving_varianceIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp!assignvariableop_39_p_re_lu_alphaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_1_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp assignvariableop_41_dense_1_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42�
Identity_43IdentityIdentity_42:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_43"#
identity_43Identity_43:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_41AssignVariableOp_412(
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
��
�,
?__inference_model_layer_call_and_return_conditional_losses_3747

inputs1
-custom_activation_sub_readvariableop_resource5
1custom_activation_truediv_readvariableop_resource3
/custom_activation_sub_1_readvariableop_resource7
3custom_activation_truediv_1_readvariableop_resource3
/custom_activation_sub_3_readvariableop_resource7
3custom_activation_truediv_3_readvariableop_resource3
/custom_activation_sub_4_readvariableop_resource7
3custom_activation_truediv_4_readvariableop_resource3
/custom_activation_sub_6_readvariableop_resource7
3custom_activation_truediv_6_readvariableop_resource3
/custom_activation_sub_7_readvariableop_resource7
3custom_activation_truediv_7_readvariableop_resource3
/custom_activation_sub_9_readvariableop_resource7
3custom_activation_truediv_9_readvariableop_resource8
4custom_activation_truediv_10_readvariableop_resource8
4custom_activation_truediv_11_readvariableop_resource8
4custom_activation_truediv_13_readvariableop_resource8
4custom_activation_truediv_14_readvariableop_resource8
4custom_activation_truediv_16_readvariableop_resource8
4custom_activation_truediv_17_readvariableop_resource8
4custom_activation_truediv_19_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource.
*batch_normalization_2_assignmovingavg_36570
,batch_normalization_2_assignmovingavg_1_3663?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource#
p_re_lu_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp�<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�$custom_activation/sub/ReadVariableOp�&custom_activation/sub_1/ReadVariableOp�'custom_activation/sub_10/ReadVariableOp�'custom_activation/sub_11/ReadVariableOp�'custom_activation/sub_12/ReadVariableOp�'custom_activation/sub_13/ReadVariableOp�'custom_activation/sub_14/ReadVariableOp�'custom_activation/sub_15/ReadVariableOp�'custom_activation/sub_16/ReadVariableOp�'custom_activation/sub_17/ReadVariableOp�'custom_activation/sub_18/ReadVariableOp�'custom_activation/sub_19/ReadVariableOp�&custom_activation/sub_2/ReadVariableOp�&custom_activation/sub_3/ReadVariableOp�&custom_activation/sub_4/ReadVariableOp�&custom_activation/sub_5/ReadVariableOp�&custom_activation/sub_6/ReadVariableOp�&custom_activation/sub_7/ReadVariableOp�&custom_activation/sub_8/ReadVariableOp�&custom_activation/sub_9/ReadVariableOp�(custom_activation/truediv/ReadVariableOp�*custom_activation/truediv_1/ReadVariableOp�+custom_activation/truediv_10/ReadVariableOp�+custom_activation/truediv_11/ReadVariableOp�+custom_activation/truediv_12/ReadVariableOp�+custom_activation/truediv_13/ReadVariableOp�+custom_activation/truediv_14/ReadVariableOp�+custom_activation/truediv_15/ReadVariableOp�+custom_activation/truediv_16/ReadVariableOp�+custom_activation/truediv_17/ReadVariableOp�+custom_activation/truediv_18/ReadVariableOp�+custom_activation/truediv_19/ReadVariableOp�*custom_activation/truediv_2/ReadVariableOp�*custom_activation/truediv_3/ReadVariableOp�*custom_activation/truediv_4/ReadVariableOp�*custom_activation/truediv_5/ReadVariableOp�*custom_activation/truediv_6/ReadVariableOp�*custom_activation/truediv_7/ReadVariableOp�*custom_activation/truediv_8/ReadVariableOp�*custom_activation/truediv_9/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�p_re_lu/ReadVariableOp�
%custom_activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2'
%custom_activation/strided_slice/stack�
'custom_activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice/stack_1�
'custom_activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2)
'custom_activation/strided_slice/stack_2�
custom_activation/strided_sliceStridedSliceinputs.custom_activation/strided_slice/stack:output:00custom_activation/strided_slice/stack_1:output:00custom_activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2!
custom_activation/strided_slice�
$custom_activation/sub/ReadVariableOpReadVariableOp-custom_activation_sub_readvariableop_resource*
_output_shapes
: *
dtype02&
$custom_activation/sub/ReadVariableOp�
custom_activation/subSub(custom_activation/strided_slice:output:0,custom_activation/sub/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub�
(custom_activation/truediv/ReadVariableOpReadVariableOp1custom_activation_truediv_readvariableop_resource*
_output_shapes
: *
dtype02*
(custom_activation/truediv/ReadVariableOp�
custom_activation/truedivRealDivcustom_activation/sub:z:00custom_activation/truediv/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv�
custom_activation/TanhTanhcustom_activation/truediv:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh�
'custom_activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_1/stack�
)custom_activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_1/stack_1�
)custom_activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_1/stack_2�
!custom_activation/strided_slice_1StridedSliceinputs0custom_activation/strided_slice_1/stack:output:02custom_activation/strided_slice_1/stack_1:output:02custom_activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_1�
&custom_activation/sub_1/ReadVariableOpReadVariableOp/custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_1/ReadVariableOp�
custom_activation/sub_1Sub*custom_activation/strided_slice_1:output:0.custom_activation/sub_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_1�
*custom_activation/truediv_1/ReadVariableOpReadVariableOp3custom_activation_truediv_1_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_1/ReadVariableOp�
custom_activation/truediv_1RealDivcustom_activation/sub_1:z:02custom_activation/truediv_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_1�
custom_activation/Tanh_1Tanhcustom_activation/truediv_1:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_1�
'custom_activation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_2/stack�
)custom_activation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_2/stack_1�
)custom_activation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_2/stack_2�
!custom_activation/strided_slice_2StridedSliceinputs0custom_activation/strided_slice_2/stack:output:02custom_activation/strided_slice_2/stack_1:output:02custom_activation/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_2�
&custom_activation/sub_2/ReadVariableOpReadVariableOp/custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_2/ReadVariableOp�
custom_activation/sub_2Sub*custom_activation/strided_slice_2:output:0.custom_activation/sub_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_2�
*custom_activation/truediv_2/ReadVariableOpReadVariableOp3custom_activation_truediv_1_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_2/ReadVariableOp�
custom_activation/truediv_2RealDivcustom_activation/sub_2:z:02custom_activation/truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_2�
custom_activation/Tanh_2Tanhcustom_activation/truediv_2:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_2�
'custom_activation/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2)
'custom_activation/strided_slice_3/stack�
)custom_activation/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_3/stack_1�
)custom_activation/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_3/stack_2�
!custom_activation/strided_slice_3StridedSliceinputs0custom_activation/strided_slice_3/stack:output:02custom_activation/strided_slice_3/stack_1:output:02custom_activation/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_3�
&custom_activation/sub_3/ReadVariableOpReadVariableOp/custom_activation_sub_3_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_3/ReadVariableOp�
custom_activation/sub_3Sub*custom_activation/strided_slice_3:output:0.custom_activation/sub_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_3�
*custom_activation/truediv_3/ReadVariableOpReadVariableOp3custom_activation_truediv_3_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_3/ReadVariableOp�
custom_activation/truediv_3RealDivcustom_activation/sub_3:z:02custom_activation/truediv_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_3�
custom_activation/Tanh_3Tanhcustom_activation/truediv_3:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_3�
'custom_activation/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_4/stack�
)custom_activation/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_4/stack_1�
)custom_activation/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_4/stack_2�
!custom_activation/strided_slice_4StridedSliceinputs0custom_activation/strided_slice_4/stack:output:02custom_activation/strided_slice_4/stack_1:output:02custom_activation/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_4�
&custom_activation/sub_4/ReadVariableOpReadVariableOp/custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_4/ReadVariableOp�
custom_activation/sub_4Sub*custom_activation/strided_slice_4:output:0.custom_activation/sub_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_4�
*custom_activation/truediv_4/ReadVariableOpReadVariableOp3custom_activation_truediv_4_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_4/ReadVariableOp�
custom_activation/truediv_4RealDivcustom_activation/sub_4:z:02custom_activation/truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_4�
custom_activation/Tanh_4Tanhcustom_activation/truediv_4:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_4�
'custom_activation/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_5/stack�
)custom_activation/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_5/stack_1�
)custom_activation/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_5/stack_2�
!custom_activation/strided_slice_5StridedSliceinputs0custom_activation/strided_slice_5/stack:output:02custom_activation/strided_slice_5/stack_1:output:02custom_activation/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_5�
&custom_activation/sub_5/ReadVariableOpReadVariableOp/custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_5/ReadVariableOp�
custom_activation/sub_5Sub*custom_activation/strided_slice_5:output:0.custom_activation/sub_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_5�
*custom_activation/truediv_5/ReadVariableOpReadVariableOp3custom_activation_truediv_4_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_5/ReadVariableOp�
custom_activation/truediv_5RealDivcustom_activation/sub_5:z:02custom_activation/truediv_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_5�
custom_activation/Tanh_5Tanhcustom_activation/truediv_5:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_5�
'custom_activation/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2)
'custom_activation/strided_slice_6/stack�
)custom_activation/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_6/stack_1�
)custom_activation/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_6/stack_2�
!custom_activation/strided_slice_6StridedSliceinputs0custom_activation/strided_slice_6/stack:output:02custom_activation/strided_slice_6/stack_1:output:02custom_activation/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_6�
&custom_activation/sub_6/ReadVariableOpReadVariableOp/custom_activation_sub_6_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_6/ReadVariableOp�
custom_activation/sub_6Sub*custom_activation/strided_slice_6:output:0.custom_activation/sub_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_6�
*custom_activation/truediv_6/ReadVariableOpReadVariableOp3custom_activation_truediv_6_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_6/ReadVariableOp�
custom_activation/truediv_6RealDivcustom_activation/sub_6:z:02custom_activation/truediv_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_6�
custom_activation/Tanh_6Tanhcustom_activation/truediv_6:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_6�
'custom_activation/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_7/stack�
)custom_activation/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_7/stack_1�
)custom_activation/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_7/stack_2�
!custom_activation/strided_slice_7StridedSliceinputs0custom_activation/strided_slice_7/stack:output:02custom_activation/strided_slice_7/stack_1:output:02custom_activation/strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_7�
&custom_activation/sub_7/ReadVariableOpReadVariableOp/custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_7/ReadVariableOp�
custom_activation/sub_7Sub*custom_activation/strided_slice_7:output:0.custom_activation/sub_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_7�
*custom_activation/truediv_7/ReadVariableOpReadVariableOp3custom_activation_truediv_7_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_7/ReadVariableOp�
custom_activation/truediv_7RealDivcustom_activation/sub_7:z:02custom_activation/truediv_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_7�
custom_activation/Tanh_7Tanhcustom_activation/truediv_7:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_7�
'custom_activation/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_8/stack�
)custom_activation/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_8/stack_1�
)custom_activation/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_8/stack_2�
!custom_activation/strided_slice_8StridedSliceinputs0custom_activation/strided_slice_8/stack:output:02custom_activation/strided_slice_8/stack_1:output:02custom_activation/strided_slice_8/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_8�
&custom_activation/sub_8/ReadVariableOpReadVariableOp/custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_8/ReadVariableOp�
custom_activation/sub_8Sub*custom_activation/strided_slice_8:output:0.custom_activation/sub_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_8�
*custom_activation/truediv_8/ReadVariableOpReadVariableOp3custom_activation_truediv_7_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_8/ReadVariableOp�
custom_activation/truediv_8RealDivcustom_activation/sub_8:z:02custom_activation/truediv_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_8�
custom_activation/Tanh_8Tanhcustom_activation/truediv_8:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_8�
'custom_activation/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2)
'custom_activation/strided_slice_9/stack�
)custom_activation/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_9/stack_1�
)custom_activation/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_9/stack_2�
!custom_activation/strided_slice_9StridedSliceinputs0custom_activation/strided_slice_9/stack:output:02custom_activation/strided_slice_9/stack_1:output:02custom_activation/strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_9�
&custom_activation/sub_9/ReadVariableOpReadVariableOp/custom_activation_sub_9_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_9/ReadVariableOp�
custom_activation/sub_9Sub*custom_activation/strided_slice_9:output:0.custom_activation/sub_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_9�
*custom_activation/truediv_9/ReadVariableOpReadVariableOp3custom_activation_truediv_9_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_9/ReadVariableOp�
custom_activation/truediv_9RealDivcustom_activation/sub_9:z:02custom_activation/truediv_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_9�
custom_activation/Tanh_9Tanhcustom_activation/truediv_9:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_9�
custom_activation/stackPackcustom_activation/Tanh:y:0custom_activation/Tanh_1:y:0custom_activation/Tanh_2:y:0custom_activation/Tanh_3:y:0custom_activation/Tanh_4:y:0custom_activation/Tanh_5:y:0custom_activation/Tanh_6:y:0custom_activation/Tanh_7:y:0custom_activation/Tanh_8:y:0custom_activation/Tanh_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2
custom_activation/stack�
(custom_activation/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_10/stack�
*custom_activation/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_10/stack_1�
*custom_activation/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_10/stack_2�
"custom_activation/strided_slice_10StridedSlice custom_activation/stack:output:01custom_activation/strided_slice_10/stack:output:03custom_activation/strided_slice_10/stack_1:output:03custom_activation/strided_slice_10/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_10�
custom_activation/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2!
custom_activation/Reshape/shape�
custom_activation/ReshapeReshape+custom_activation/strided_slice_10:output:0(custom_activation/Reshape/shape:output:0*
T0*+
_output_shapes
:���������
2
custom_activation/Reshape�
(custom_activation/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_11/stack�
*custom_activation/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_11/stack_1�
*custom_activation/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_11/stack_2�
"custom_activation/strided_slice_11StridedSlice custom_activation/stack:output:01custom_activation/strided_slice_11/stack:output:03custom_activation/strided_slice_11/stack_1:output:03custom_activation/strided_slice_11/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_11�
!custom_activation/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2#
!custom_activation/Reshape_1/shape�
custom_activation/Reshape_1Reshape+custom_activation/strided_slice_11:output:0*custom_activation/Reshape_1/shape:output:0*
T0*+
_output_shapes
:���������
2
custom_activation/Reshape_1�
custom_activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
custom_activation/concat/axis�
custom_activation/concatConcatV2$custom_activation/Reshape_1:output:0 custom_activation/stack:output:0"custom_activation/Reshape:output:0&custom_activation/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
2
custom_activation/concat�
!custom_activation/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2#
!custom_activation/Reshape_2/shape�
custom_activation/Reshape_2Reshape!custom_activation/concat:output:0*custom_activation/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������
2
custom_activation/Reshape_2�
(custom_activation/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_12/stack�
*custom_activation/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_12/stack_1�
*custom_activation/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_12/stack_2�
"custom_activation/strided_slice_12StridedSliceinputs1custom_activation/strided_slice_12/stack:output:03custom_activation/strided_slice_12/stack_1:output:03custom_activation/strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_12�
'custom_activation/sub_10/ReadVariableOpReadVariableOp-custom_activation_sub_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_10/ReadVariableOp�
custom_activation/sub_10Sub+custom_activation/strided_slice_12:output:0/custom_activation/sub_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_10�
+custom_activation/truediv_10/ReadVariableOpReadVariableOp4custom_activation_truediv_10_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_10/ReadVariableOp�
custom_activation/truediv_10RealDivcustom_activation/sub_10:z:03custom_activation/truediv_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_10�
custom_activation/ReluRelu custom_activation/truediv_10:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu�
custom_activation/Log1pLog1p$custom_activation/Relu:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p�
(custom_activation/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_13/stack�
*custom_activation/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_13/stack_1�
*custom_activation/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_13/stack_2�
"custom_activation/strided_slice_13StridedSliceinputs1custom_activation/strided_slice_13/stack:output:03custom_activation/strided_slice_13/stack_1:output:03custom_activation/strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_13�
'custom_activation/sub_11/ReadVariableOpReadVariableOp/custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_11/ReadVariableOp�
custom_activation/sub_11Sub+custom_activation/strided_slice_13:output:0/custom_activation/sub_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_11�
+custom_activation/truediv_11/ReadVariableOpReadVariableOp4custom_activation_truediv_11_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_11/ReadVariableOp�
custom_activation/truediv_11RealDivcustom_activation/sub_11:z:03custom_activation/truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_11�
custom_activation/Relu_1Relu custom_activation/truediv_11:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_1�
custom_activation/Log1p_1Log1p&custom_activation/Relu_1:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_1�
(custom_activation/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_14/stack�
*custom_activation/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_14/stack_1�
*custom_activation/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_14/stack_2�
"custom_activation/strided_slice_14StridedSliceinputs1custom_activation/strided_slice_14/stack:output:03custom_activation/strided_slice_14/stack_1:output:03custom_activation/strided_slice_14/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_14�
'custom_activation/sub_12/ReadVariableOpReadVariableOp/custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_12/ReadVariableOp�
custom_activation/sub_12Sub+custom_activation/strided_slice_14:output:0/custom_activation/sub_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_12�
+custom_activation/truediv_12/ReadVariableOpReadVariableOp4custom_activation_truediv_11_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_12/ReadVariableOp�
custom_activation/truediv_12RealDivcustom_activation/sub_12:z:03custom_activation/truediv_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_12�
custom_activation/Relu_2Relu custom_activation/truediv_12:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_2�
custom_activation/Log1p_2Log1p&custom_activation/Relu_2:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_2�
(custom_activation/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_15/stack�
*custom_activation/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_15/stack_1�
*custom_activation/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_15/stack_2�
"custom_activation/strided_slice_15StridedSliceinputs1custom_activation/strided_slice_15/stack:output:03custom_activation/strided_slice_15/stack_1:output:03custom_activation/strided_slice_15/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_15�
'custom_activation/sub_13/ReadVariableOpReadVariableOp/custom_activation_sub_3_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_13/ReadVariableOp�
custom_activation/sub_13Sub+custom_activation/strided_slice_15:output:0/custom_activation/sub_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_13�
+custom_activation/truediv_13/ReadVariableOpReadVariableOp4custom_activation_truediv_13_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_13/ReadVariableOp�
custom_activation/truediv_13RealDivcustom_activation/sub_13:z:03custom_activation/truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_13�
custom_activation/Relu_3Relu custom_activation/truediv_13:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_3�
custom_activation/Log1p_3Log1p&custom_activation/Relu_3:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_3�
(custom_activation/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_16/stack�
*custom_activation/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_16/stack_1�
*custom_activation/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_16/stack_2�
"custom_activation/strided_slice_16StridedSliceinputs1custom_activation/strided_slice_16/stack:output:03custom_activation/strided_slice_16/stack_1:output:03custom_activation/strided_slice_16/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_16�
'custom_activation/sub_14/ReadVariableOpReadVariableOp/custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_14/ReadVariableOp�
custom_activation/sub_14Sub+custom_activation/strided_slice_16:output:0/custom_activation/sub_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_14�
+custom_activation/truediv_14/ReadVariableOpReadVariableOp4custom_activation_truediv_14_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_14/ReadVariableOp�
custom_activation/truediv_14RealDivcustom_activation/sub_14:z:03custom_activation/truediv_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_14�
custom_activation/Relu_4Relu custom_activation/truediv_14:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_4�
custom_activation/Log1p_4Log1p&custom_activation/Relu_4:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_4�
(custom_activation/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_17/stack�
*custom_activation/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_17/stack_1�
*custom_activation/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_17/stack_2�
"custom_activation/strided_slice_17StridedSliceinputs1custom_activation/strided_slice_17/stack:output:03custom_activation/strided_slice_17/stack_1:output:03custom_activation/strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_17�
'custom_activation/sub_15/ReadVariableOpReadVariableOp/custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_15/ReadVariableOp�
custom_activation/sub_15Sub+custom_activation/strided_slice_17:output:0/custom_activation/sub_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_15�
+custom_activation/truediv_15/ReadVariableOpReadVariableOp4custom_activation_truediv_14_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_15/ReadVariableOp�
custom_activation/truediv_15RealDivcustom_activation/sub_15:z:03custom_activation/truediv_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_15�
custom_activation/Relu_5Relu custom_activation/truediv_15:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_5�
custom_activation/Log1p_5Log1p&custom_activation/Relu_5:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_5�
(custom_activation/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_18/stack�
*custom_activation/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_18/stack_1�
*custom_activation/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_18/stack_2�
"custom_activation/strided_slice_18StridedSliceinputs1custom_activation/strided_slice_18/stack:output:03custom_activation/strided_slice_18/stack_1:output:03custom_activation/strided_slice_18/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_18�
'custom_activation/sub_16/ReadVariableOpReadVariableOp/custom_activation_sub_6_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_16/ReadVariableOp�
custom_activation/sub_16Sub+custom_activation/strided_slice_18:output:0/custom_activation/sub_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_16�
+custom_activation/truediv_16/ReadVariableOpReadVariableOp4custom_activation_truediv_16_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_16/ReadVariableOp�
custom_activation/truediv_16RealDivcustom_activation/sub_16:z:03custom_activation/truediv_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_16�
custom_activation/Relu_6Relu custom_activation/truediv_16:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_6�
custom_activation/Log1p_6Log1p&custom_activation/Relu_6:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_6�
(custom_activation/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_19/stack�
*custom_activation/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_19/stack_1�
*custom_activation/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_19/stack_2�
"custom_activation/strided_slice_19StridedSliceinputs1custom_activation/strided_slice_19/stack:output:03custom_activation/strided_slice_19/stack_1:output:03custom_activation/strided_slice_19/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_19�
'custom_activation/sub_17/ReadVariableOpReadVariableOp/custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_17/ReadVariableOp�
custom_activation/sub_17Sub+custom_activation/strided_slice_19:output:0/custom_activation/sub_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_17�
+custom_activation/truediv_17/ReadVariableOpReadVariableOp4custom_activation_truediv_17_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_17/ReadVariableOp�
custom_activation/truediv_17RealDivcustom_activation/sub_17:z:03custom_activation/truediv_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_17�
custom_activation/Relu_7Relu custom_activation/truediv_17:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_7�
custom_activation/Log1p_7Log1p&custom_activation/Relu_7:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_7�
(custom_activation/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_20/stack�
*custom_activation/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_20/stack_1�
*custom_activation/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_20/stack_2�
"custom_activation/strided_slice_20StridedSliceinputs1custom_activation/strided_slice_20/stack:output:03custom_activation/strided_slice_20/stack_1:output:03custom_activation/strided_slice_20/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_20�
'custom_activation/sub_18/ReadVariableOpReadVariableOp/custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_18/ReadVariableOp�
custom_activation/sub_18Sub+custom_activation/strided_slice_20:output:0/custom_activation/sub_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_18�
+custom_activation/truediv_18/ReadVariableOpReadVariableOp4custom_activation_truediv_17_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_18/ReadVariableOp�
custom_activation/truediv_18RealDivcustom_activation/sub_18:z:03custom_activation/truediv_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_18�
custom_activation/Relu_8Relu custom_activation/truediv_18:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_8�
custom_activation/Log1p_8Log1p&custom_activation/Relu_8:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_8�
(custom_activation/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_21/stack�
*custom_activation/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_21/stack_1�
*custom_activation/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_21/stack_2�
"custom_activation/strided_slice_21StridedSliceinputs1custom_activation/strided_slice_21/stack:output:03custom_activation/strided_slice_21/stack_1:output:03custom_activation/strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_21�
'custom_activation/sub_19/ReadVariableOpReadVariableOp/custom_activation_sub_9_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_19/ReadVariableOp�
custom_activation/sub_19Sub+custom_activation/strided_slice_21:output:0/custom_activation/sub_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_19�
+custom_activation/truediv_19/ReadVariableOpReadVariableOp4custom_activation_truediv_19_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_19/ReadVariableOp�
custom_activation/truediv_19RealDivcustom_activation/sub_19:z:03custom_activation/truediv_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_19�
custom_activation/Relu_9Relu custom_activation/truediv_19:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_9�
custom_activation/Log1p_9Log1p&custom_activation/Relu_9:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_9�
custom_activation/stack_1Packcustom_activation/Log1p:y:0custom_activation/Log1p_1:y:0custom_activation/Log1p_2:y:0custom_activation/Log1p_3:y:0custom_activation/Log1p_4:y:0custom_activation/Log1p_5:y:0custom_activation/Log1p_6:y:0custom_activation/Log1p_7:y:0custom_activation/Log1p_8:y:0custom_activation/Log1p_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2
custom_activation/stack_1�
(custom_activation/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_22/stack�
*custom_activation/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_22/stack_1�
*custom_activation/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_22/stack_2�
"custom_activation/strided_slice_22StridedSlice"custom_activation/stack_1:output:01custom_activation/strided_slice_22/stack:output:03custom_activation/strided_slice_22/stack_1:output:03custom_activation/strided_slice_22/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_22�
!custom_activation/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2#
!custom_activation/Reshape_3/shape�
custom_activation/Reshape_3Reshape+custom_activation/strided_slice_22:output:0*custom_activation/Reshape_3/shape:output:0*
T0*+
_output_shapes
:���������
2
custom_activation/Reshape_3�
(custom_activation/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_23/stack�
*custom_activation/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_23/stack_1�
*custom_activation/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_23/stack_2�
"custom_activation/strided_slice_23StridedSlice"custom_activation/stack_1:output:01custom_activation/strided_slice_23/stack:output:03custom_activation/strided_slice_23/stack_1:output:03custom_activation/strided_slice_23/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_23�
!custom_activation/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2#
!custom_activation/Reshape_4/shape�
custom_activation/Reshape_4Reshape+custom_activation/strided_slice_23:output:0*custom_activation/Reshape_4/shape:output:0*
T0*+
_output_shapes
:���������
2
custom_activation/Reshape_4�
custom_activation/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
custom_activation/concat_1/axis�
custom_activation/concat_1ConcatV2$custom_activation/Reshape_4:output:0"custom_activation/stack_1:output:0$custom_activation/Reshape_3:output:0(custom_activation/concat_1/axis:output:0*
N*
T0*+
_output_shapes
:���������
2
custom_activation/concat_1�
!custom_activation/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2#
!custom_activation/Reshape_5/shape�
custom_activation/Reshape_5Reshape#custom_activation/concat_1:output:0*custom_activation/Reshape_5/shape:output:0*
T0*/
_output_shapes
:���������
2
custom_activation/Reshape_5�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2D$custom_activation/Reshape_5:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2d_1/BiasAdd�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2D$custom_activation/Reshape_2:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2d/BiasAdd�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_1/FusedBatchNormV3�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2&
$batch_normalization/FusedBatchNormV3�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1�
tf.math.tanh_1/TanhTanh*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
tf.math.tanh_1/Tanh�
tf.nn.relu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
tf.nn.relu_1/Relu�
tf.math.tanh/TanhTanh(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
tf.math.tanh/Tanh�
tf.nn.relu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
tf.nn.relu/Relu�
average_pooling2d_3/AvgPoolAvgPooltf.math.tanh_1/Tanh:y:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_3/AvgPool�
max_pooling2d_3/MaxPoolMaxPooltf.math.tanh_1/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool�
average_pooling2d_2/AvgPoolAvgPooltf.nn.relu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_2/AvgPool�
max_pooling2d_2/MaxPoolMaxPooltf.nn.relu_1/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool�
average_pooling2d_1/AvgPoolAvgPooltf.math.tanh/Tanh:y:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool�
max_pooling2d_1/MaxPoolMaxPooltf.math.tanh/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool�
average_pooling2d/AvgPoolAvgPooltf.nn.relu/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool�
max_pooling2d/MaxPoolMaxPooltf.nn.relu/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_1/Const�
flatten_1/ReshapeReshape"average_pooling2d/AvgPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_2/Const�
flatten_2/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_2/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_3/Const�
flatten_3/ReshapeReshape$average_pooling2d_1/AvgPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_3/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_4/Const�
flatten_4/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_4/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_5/Const�
flatten_5/ReshapeReshape$average_pooling2d_2/AvgPool:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_5/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_6/Const�
flatten_6/ReshapeReshape max_pooling2d_3/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_6/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_7/Const�
flatten_7/ReshapeReshape$average_pooling2d_3/AvgPool:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_7/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0flatten_5/Reshape:output:0flatten_6/Reshape:output:0flatten_7/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate/concat�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAdd�
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indices�
"batch_normalization_2/moments/meanMeandense/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_2/moments/mean�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_2/moments/StopGradient�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_2/moments/SquaredDifference�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices�
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_2/moments/variance�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze�
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1�
+batch_normalization_2/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/3657*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_2/AssignMovingAvg/decay�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_2_assignmovingavg_3657*
_output_shapes	
:�*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/3657*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/sub�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/3657*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/mul�
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_2_assignmovingavg_3657-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/3657*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_2/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/3663*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/AssignMovingAvg_1/decay�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_1_3663*
_output_shapes	
:�*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/3663*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/sub�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/3663*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/mul�
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_1_3663/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/3663*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp�
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_2/batchnorm/add/y�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/mul_1�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/add_1�
p_re_lu/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
p_re_lu/Relu�
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*
_output_shapes	
:�*
dtype02
p_re_lu/ReadVariableOpg
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
p_re_lu/Neg�
p_re_lu/Neg_1Neg)batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
p_re_lu/Neg_1n
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*(
_output_shapes
:����������2
p_re_lu/Relu_1�
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*(
_output_shapes
:����������2
p_re_lu/mul�
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*(
_output_shapes
:����������2
p_re_lu/adds
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const�
dropout/dropout/MulMulp_re_lu/add:z:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/dropout/Mulm
dropout/dropout/ShapeShapep_re_lu/add:z:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Softmax�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentitydense_1/Softmax:softmax:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp%^custom_activation/sub/ReadVariableOp'^custom_activation/sub_1/ReadVariableOp(^custom_activation/sub_10/ReadVariableOp(^custom_activation/sub_11/ReadVariableOp(^custom_activation/sub_12/ReadVariableOp(^custom_activation/sub_13/ReadVariableOp(^custom_activation/sub_14/ReadVariableOp(^custom_activation/sub_15/ReadVariableOp(^custom_activation/sub_16/ReadVariableOp(^custom_activation/sub_17/ReadVariableOp(^custom_activation/sub_18/ReadVariableOp(^custom_activation/sub_19/ReadVariableOp'^custom_activation/sub_2/ReadVariableOp'^custom_activation/sub_3/ReadVariableOp'^custom_activation/sub_4/ReadVariableOp'^custom_activation/sub_5/ReadVariableOp'^custom_activation/sub_6/ReadVariableOp'^custom_activation/sub_7/ReadVariableOp'^custom_activation/sub_8/ReadVariableOp'^custom_activation/sub_9/ReadVariableOp)^custom_activation/truediv/ReadVariableOp+^custom_activation/truediv_1/ReadVariableOp,^custom_activation/truediv_10/ReadVariableOp,^custom_activation/truediv_11/ReadVariableOp,^custom_activation/truediv_12/ReadVariableOp,^custom_activation/truediv_13/ReadVariableOp,^custom_activation/truediv_14/ReadVariableOp,^custom_activation/truediv_15/ReadVariableOp,^custom_activation/truediv_16/ReadVariableOp,^custom_activation/truediv_17/ReadVariableOp,^custom_activation/truediv_18/ReadVariableOp,^custom_activation/truediv_19/ReadVariableOp+^custom_activation/truediv_2/ReadVariableOp+^custom_activation/truediv_3/ReadVariableOp+^custom_activation/truediv_4/ReadVariableOp+^custom_activation/truediv_5/ReadVariableOp+^custom_activation/truediv_6/ReadVariableOp+^custom_activation/truediv_7/ReadVariableOp+^custom_activation/truediv_8/ReadVariableOp+^custom_activation/truediv_9/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^p_re_lu/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2L
$custom_activation/sub/ReadVariableOp$custom_activation/sub/ReadVariableOp2P
&custom_activation/sub_1/ReadVariableOp&custom_activation/sub_1/ReadVariableOp2R
'custom_activation/sub_10/ReadVariableOp'custom_activation/sub_10/ReadVariableOp2R
'custom_activation/sub_11/ReadVariableOp'custom_activation/sub_11/ReadVariableOp2R
'custom_activation/sub_12/ReadVariableOp'custom_activation/sub_12/ReadVariableOp2R
'custom_activation/sub_13/ReadVariableOp'custom_activation/sub_13/ReadVariableOp2R
'custom_activation/sub_14/ReadVariableOp'custom_activation/sub_14/ReadVariableOp2R
'custom_activation/sub_15/ReadVariableOp'custom_activation/sub_15/ReadVariableOp2R
'custom_activation/sub_16/ReadVariableOp'custom_activation/sub_16/ReadVariableOp2R
'custom_activation/sub_17/ReadVariableOp'custom_activation/sub_17/ReadVariableOp2R
'custom_activation/sub_18/ReadVariableOp'custom_activation/sub_18/ReadVariableOp2R
'custom_activation/sub_19/ReadVariableOp'custom_activation/sub_19/ReadVariableOp2P
&custom_activation/sub_2/ReadVariableOp&custom_activation/sub_2/ReadVariableOp2P
&custom_activation/sub_3/ReadVariableOp&custom_activation/sub_3/ReadVariableOp2P
&custom_activation/sub_4/ReadVariableOp&custom_activation/sub_4/ReadVariableOp2P
&custom_activation/sub_5/ReadVariableOp&custom_activation/sub_5/ReadVariableOp2P
&custom_activation/sub_6/ReadVariableOp&custom_activation/sub_6/ReadVariableOp2P
&custom_activation/sub_7/ReadVariableOp&custom_activation/sub_7/ReadVariableOp2P
&custom_activation/sub_8/ReadVariableOp&custom_activation/sub_8/ReadVariableOp2P
&custom_activation/sub_9/ReadVariableOp&custom_activation/sub_9/ReadVariableOp2T
(custom_activation/truediv/ReadVariableOp(custom_activation/truediv/ReadVariableOp2X
*custom_activation/truediv_1/ReadVariableOp*custom_activation/truediv_1/ReadVariableOp2Z
+custom_activation/truediv_10/ReadVariableOp+custom_activation/truediv_10/ReadVariableOp2Z
+custom_activation/truediv_11/ReadVariableOp+custom_activation/truediv_11/ReadVariableOp2Z
+custom_activation/truediv_12/ReadVariableOp+custom_activation/truediv_12/ReadVariableOp2Z
+custom_activation/truediv_13/ReadVariableOp+custom_activation/truediv_13/ReadVariableOp2Z
+custom_activation/truediv_14/ReadVariableOp+custom_activation/truediv_14/ReadVariableOp2Z
+custom_activation/truediv_15/ReadVariableOp+custom_activation/truediv_15/ReadVariableOp2Z
+custom_activation/truediv_16/ReadVariableOp+custom_activation/truediv_16/ReadVariableOp2Z
+custom_activation/truediv_17/ReadVariableOp+custom_activation/truediv_17/ReadVariableOp2Z
+custom_activation/truediv_18/ReadVariableOp+custom_activation/truediv_18/ReadVariableOp2Z
+custom_activation/truediv_19/ReadVariableOp+custom_activation/truediv_19/ReadVariableOp2X
*custom_activation/truediv_2/ReadVariableOp*custom_activation/truediv_2/ReadVariableOp2X
*custom_activation/truediv_3/ReadVariableOp*custom_activation/truediv_3/ReadVariableOp2X
*custom_activation/truediv_4/ReadVariableOp*custom_activation/truediv_4/ReadVariableOp2X
*custom_activation/truediv_5/ReadVariableOp*custom_activation/truediv_5/ReadVariableOp2X
*custom_activation/truediv_6/ReadVariableOp*custom_activation/truediv_6/ReadVariableOp2X
*custom_activation/truediv_7/ReadVariableOp*custom_activation/truediv_7/ReadVariableOp2X
*custom_activation/truediv_8/ReadVariableOp*custom_activation/truediv_8/ReadVariableOp2X
*custom_activation/truediv_9/ReadVariableOp*custom_activation/truediv_9/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp20
p_re_lu/ReadVariableOpp_re_lu/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_flatten_6_layer_call_fn_5114

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
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_6_layer_call_and_return_conditional_losses_23032
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1379

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�R
�
__inference__traced_save_5572
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_4_read_readvariableop)
%savev2_variable_5_read_readvariableop)
%savev2_variable_6_read_readvariableop)
%savev2_variable_7_read_readvariableop)
%savev2_variable_8_read_readvariableop)
%savev2_variable_9_read_readvariableop*
&savev2_variable_10_read_readvariableop*
&savev2_variable_11_read_readvariableop*
&savev2_variable_12_read_readvariableop*
&savev2_variable_13_read_readvariableop*
&savev2_variable_14_read_readvariableop*
&savev2_variable_15_read_readvariableop*
&savev2_variable_16_read_readvariableop*
&savev2_variable_17_read_readvariableop*
&savev2_variable_18_read_readvariableop*
&savev2_variable_19_read_readvariableop*
&savev2_variable_20_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop,
(savev2_p_re_lu_alpha_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/mu_t_1/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_t_2/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_t_3/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_t_4/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_1/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_2/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_3/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_4/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_5/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_6/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_7/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_t_8/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_p_1/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_p_2/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/mu_p_3/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_1/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_2/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_3/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_4/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_5/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-0/sigma_p_6/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_4_read_readvariableop%savev2_variable_5_read_readvariableop%savev2_variable_6_read_readvariableop%savev2_variable_7_read_readvariableop%savev2_variable_8_read_readvariableop%savev2_variable_9_read_readvariableop&savev2_variable_10_read_readvariableop&savev2_variable_11_read_readvariableop&savev2_variable_12_read_readvariableop&savev2_variable_13_read_readvariableop&savev2_variable_14_read_readvariableop&savev2_variable_15_read_readvariableop&savev2_variable_16_read_readvariableop&savev2_variable_17_read_readvariableop&savev2_variable_18_read_readvariableop&savev2_variable_19_read_readvariableop&savev2_variable_20_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop(savev2_p_re_lu_alpha_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+2
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : :::::::::::::
��:�:�:�:�:�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::&""
 
_output_shapes
:
��:!#

_output_shapes	
:�:!$

_output_shapes	
:�:!%

_output_shapes	
:�:!&

_output_shapes	
:�:!'

_output_shapes	
:�:!(

_output_shapes	
:�:%)!

_output_shapes
:	�: *

_output_shapes
::+

_output_shapes
: 
��
�
K__inference_custom_activation_layer_call_and_return_conditional_losses_1860

inputs
sub_readvariableop_resource#
truediv_readvariableop_resource!
sub_1_readvariableop_resource%
!truediv_1_readvariableop_resource!
sub_3_readvariableop_resource%
!truediv_3_readvariableop_resource!
sub_4_readvariableop_resource%
!truediv_4_readvariableop_resource!
sub_6_readvariableop_resource%
!truediv_6_readvariableop_resource!
sub_7_readvariableop_resource%
!truediv_7_readvariableop_resource!
sub_9_readvariableop_resource%
!truediv_9_readvariableop_resource&
"truediv_10_readvariableop_resource&
"truediv_11_readvariableop_resource&
"truediv_13_readvariableop_resource&
"truediv_14_readvariableop_resource&
"truediv_16_readvariableop_resource&
"truediv_17_readvariableop_resource&
"truediv_19_readvariableop_resource
identity

identity_1��sub/ReadVariableOp�sub_1/ReadVariableOp�sub_10/ReadVariableOp�sub_11/ReadVariableOp�sub_12/ReadVariableOp�sub_13/ReadVariableOp�sub_14/ReadVariableOp�sub_15/ReadVariableOp�sub_16/ReadVariableOp�sub_17/ReadVariableOp�sub_18/ReadVariableOp�sub_19/ReadVariableOp�sub_2/ReadVariableOp�sub_3/ReadVariableOp�sub_4/ReadVariableOp�sub_5/ReadVariableOp�sub_6/ReadVariableOp�sub_7/ReadVariableOp�sub_8/ReadVariableOp�sub_9/ReadVariableOp�truediv/ReadVariableOp�truediv_1/ReadVariableOp�truediv_10/ReadVariableOp�truediv_11/ReadVariableOp�truediv_12/ReadVariableOp�truediv_13/ReadVariableOp�truediv_14/ReadVariableOp�truediv_15/ReadVariableOp�truediv_16/ReadVariableOp�truediv_17/ReadVariableOp�truediv_18/ReadVariableOp�truediv_19/ReadVariableOp�truediv_2/ReadVariableOp�truediv_3/ReadVariableOp�truediv_4/ReadVariableOp�truediv_5/ReadVariableOp�truediv_6/ReadVariableOp�truediv_7/ReadVariableOp�truediv_8/ReadVariableOp�truediv_9/ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice|
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
: *
dtype02
sub/ReadVariableOpw
subSubstrided_slice:output:0sub/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub�
truediv/ReadVariableOpReadVariableOptruediv_readvariableop_resource*
_output_shapes
: *
dtype02
truediv/ReadVariableOpx
truedivRealDivsub:z:0truediv/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
truedivS
TanhTanhtruediv:z:0*
T0*'
_output_shapes
:���������2
Tanh�
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1�
sub_1/ReadVariableOpReadVariableOpsub_1_readvariableop_resource*
_output_shapes
: *
dtype02
sub_1/ReadVariableOp
sub_1Substrided_slice_1:output:0sub_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_1�
truediv_1/ReadVariableOpReadVariableOp!truediv_1_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_1/ReadVariableOp�
	truediv_1RealDiv	sub_1:z:0 truediv_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_1Y
Tanh_1Tanhtruediv_1:z:0*
T0*'
_output_shapes
:���������2
Tanh_1�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2�
sub_2/ReadVariableOpReadVariableOpsub_1_readvariableop_resource*
_output_shapes
: *
dtype02
sub_2/ReadVariableOp
sub_2Substrided_slice_2:output:0sub_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_2�
truediv_2/ReadVariableOpReadVariableOp!truediv_1_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_2/ReadVariableOp�
	truediv_2RealDiv	sub_2:z:0 truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_2Y
Tanh_2Tanhtruediv_2:z:0*
T0*'
_output_shapes
:���������2
Tanh_2�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3�
sub_3/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
: *
dtype02
sub_3/ReadVariableOp
sub_3Substrided_slice_3:output:0sub_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_3�
truediv_3/ReadVariableOpReadVariableOp!truediv_3_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_3/ReadVariableOp�
	truediv_3RealDiv	sub_3:z:0 truediv_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_3Y
Tanh_3Tanhtruediv_3:z:0*
T0*'
_output_shapes
:���������2
Tanh_3�
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_4/stack_2�
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
sub_4/ReadVariableOpReadVariableOpsub_4_readvariableop_resource*
_output_shapes
: *
dtype02
sub_4/ReadVariableOp
sub_4Substrided_slice_4:output:0sub_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_4�
truediv_4/ReadVariableOpReadVariableOp!truediv_4_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_4/ReadVariableOp�
	truediv_4RealDiv	sub_4:z:0 truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_4Y
Tanh_4Tanhtruediv_4:z:0*
T0*'
_output_shapes
:���������2
Tanh_4�
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_5/stack�
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_5/stack_1�
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_5/stack_2�
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5�
sub_5/ReadVariableOpReadVariableOpsub_4_readvariableop_resource*
_output_shapes
: *
dtype02
sub_5/ReadVariableOp
sub_5Substrided_slice_5:output:0sub_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_5�
truediv_5/ReadVariableOpReadVariableOp!truediv_4_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_5/ReadVariableOp�
	truediv_5RealDiv	sub_5:z:0 truediv_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_5Y
Tanh_5Tanhtruediv_5:z:0*
T0*'
_output_shapes
:���������2
Tanh_5�
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_6/stack�
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_6/stack_1�
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_6/stack_2�
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6�
sub_6/ReadVariableOpReadVariableOpsub_6_readvariableop_resource*
_output_shapes
: *
dtype02
sub_6/ReadVariableOp
sub_6Substrided_slice_6:output:0sub_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_6�
truediv_6/ReadVariableOpReadVariableOp!truediv_6_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_6/ReadVariableOp�
	truediv_6RealDiv	sub_6:z:0 truediv_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_6Y
Tanh_6Tanhtruediv_6:z:0*
T0*'
_output_shapes
:���������2
Tanh_6�
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_7/stack�
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_7/stack_1�
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_7/stack_2�
strided_slice_7StridedSliceinputsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7�
sub_7/ReadVariableOpReadVariableOpsub_7_readvariableop_resource*
_output_shapes
: *
dtype02
sub_7/ReadVariableOp
sub_7Substrided_slice_7:output:0sub_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_7�
truediv_7/ReadVariableOpReadVariableOp!truediv_7_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_7/ReadVariableOp�
	truediv_7RealDiv	sub_7:z:0 truediv_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_7Y
Tanh_7Tanhtruediv_7:z:0*
T0*'
_output_shapes
:���������2
Tanh_7�
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_8/stack�
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_8/stack_1�
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_8/stack_2�
strided_slice_8StridedSliceinputsstrided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8�
sub_8/ReadVariableOpReadVariableOpsub_7_readvariableop_resource*
_output_shapes
: *
dtype02
sub_8/ReadVariableOp
sub_8Substrided_slice_8:output:0sub_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_8�
truediv_8/ReadVariableOpReadVariableOp!truediv_7_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_8/ReadVariableOp�
	truediv_8RealDiv	sub_8:z:0 truediv_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_8Y
Tanh_8Tanhtruediv_8:z:0*
T0*'
_output_shapes
:���������2
Tanh_8�
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_9/stack�
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_9/stack_1�
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_9/stack_2�
strided_slice_9StridedSliceinputsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9�
sub_9/ReadVariableOpReadVariableOpsub_9_readvariableop_resource*
_output_shapes
: *
dtype02
sub_9/ReadVariableOp
sub_9Substrided_slice_9:output:0sub_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_9�
truediv_9/ReadVariableOpReadVariableOp!truediv_9_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_9/ReadVariableOp�
	truediv_9RealDiv	sub_9:z:0 truediv_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	truediv_9Y
Tanh_9Tanhtruediv_9:z:0*
T0*'
_output_shapes
:���������2
Tanh_9�
stackPackTanh:y:0
Tanh_1:y:0
Tanh_2:y:0
Tanh_3:y:0
Tanh_4:y:0
Tanh_5:y:0
Tanh_6:y:0
Tanh_7:y:0
Tanh_8:y:0
Tanh_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2
stack�
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_10/stack�
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_10/stack_1�
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_10/stack_2�
strided_slice_10StridedSlicestack:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10s
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2
Reshape/shape�
ReshapeReshapestrided_slice_10:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:���������
2	
Reshape�
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_11/stack�
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_11/stack_1�
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_11/stack_2�
strided_slice_11StridedSlicestack:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11w
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2
Reshape_1/shape�
	Reshape_1Reshapestrided_slice_11:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:���������
2
	Reshape_1e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2Reshape_1:output:0stack:output:0Reshape:output:0concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
2
concat{
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2
Reshape_2/shape�
	Reshape_2Reshapeconcat:output:0Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������
2
	Reshape_2�
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_12/stack�
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_12/stack_1�
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_12/stack_2�
strided_slice_12StridedSliceinputsstrided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12�
sub_10/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
: *
dtype02
sub_10/ReadVariableOp�
sub_10Substrided_slice_12:output:0sub_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_10�
truediv_10/ReadVariableOpReadVariableOp"truediv_10_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_10/ReadVariableOp�

truediv_10RealDiv
sub_10:z:0!truediv_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_10V
ReluRelutruediv_10:z:0*
T0*'
_output_shapes
:���������2
Relu]
Log1pLog1pRelu:activations:0*
T0*'
_output_shapes
:���������2
Log1p�
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_13/stack�
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_13/stack_1�
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_13/stack_2�
strided_slice_13StridedSliceinputsstrided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13�
sub_11/ReadVariableOpReadVariableOpsub_1_readvariableop_resource*
_output_shapes
: *
dtype02
sub_11/ReadVariableOp�
sub_11Substrided_slice_13:output:0sub_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_11�
truediv_11/ReadVariableOpReadVariableOp"truediv_11_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_11/ReadVariableOp�

truediv_11RealDiv
sub_11:z:0!truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_11Z
Relu_1Relutruediv_11:z:0*
T0*'
_output_shapes
:���������2
Relu_1c
Log1p_1Log1pRelu_1:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_1�
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_14/stack�
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_14/stack_1�
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_14/stack_2�
strided_slice_14StridedSliceinputsstrided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14�
sub_12/ReadVariableOpReadVariableOpsub_1_readvariableop_resource*
_output_shapes
: *
dtype02
sub_12/ReadVariableOp�
sub_12Substrided_slice_14:output:0sub_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_12�
truediv_12/ReadVariableOpReadVariableOp"truediv_11_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_12/ReadVariableOp�

truediv_12RealDiv
sub_12:z:0!truediv_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_12Z
Relu_2Relutruediv_12:z:0*
T0*'
_output_shapes
:���������2
Relu_2c
Log1p_2Log1pRelu_2:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_2�
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_15/stack�
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_15/stack_1�
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_15/stack_2�
strided_slice_15StridedSliceinputsstrided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15�
sub_13/ReadVariableOpReadVariableOpsub_3_readvariableop_resource*
_output_shapes
: *
dtype02
sub_13/ReadVariableOp�
sub_13Substrided_slice_15:output:0sub_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_13�
truediv_13/ReadVariableOpReadVariableOp"truediv_13_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_13/ReadVariableOp�

truediv_13RealDiv
sub_13:z:0!truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_13Z
Relu_3Relutruediv_13:z:0*
T0*'
_output_shapes
:���������2
Relu_3c
Log1p_3Log1pRelu_3:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_3�
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_16/stack�
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_16/stack_1�
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_16/stack_2�
strided_slice_16StridedSliceinputsstrided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16�
sub_14/ReadVariableOpReadVariableOpsub_4_readvariableop_resource*
_output_shapes
: *
dtype02
sub_14/ReadVariableOp�
sub_14Substrided_slice_16:output:0sub_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_14�
truediv_14/ReadVariableOpReadVariableOp"truediv_14_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_14/ReadVariableOp�

truediv_14RealDiv
sub_14:z:0!truediv_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_14Z
Relu_4Relutruediv_14:z:0*
T0*'
_output_shapes
:���������2
Relu_4c
Log1p_4Log1pRelu_4:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_4�
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_17/stack�
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_17/stack_1�
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_17/stack_2�
strided_slice_17StridedSliceinputsstrided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17�
sub_15/ReadVariableOpReadVariableOpsub_4_readvariableop_resource*
_output_shapes
: *
dtype02
sub_15/ReadVariableOp�
sub_15Substrided_slice_17:output:0sub_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_15�
truediv_15/ReadVariableOpReadVariableOp"truediv_14_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_15/ReadVariableOp�

truediv_15RealDiv
sub_15:z:0!truediv_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_15Z
Relu_5Relutruediv_15:z:0*
T0*'
_output_shapes
:���������2
Relu_5c
Log1p_5Log1pRelu_5:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_5�
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_18/stack�
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_18/stack_1�
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_18/stack_2�
strided_slice_18StridedSliceinputsstrided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18�
sub_16/ReadVariableOpReadVariableOpsub_6_readvariableop_resource*
_output_shapes
: *
dtype02
sub_16/ReadVariableOp�
sub_16Substrided_slice_18:output:0sub_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_16�
truediv_16/ReadVariableOpReadVariableOp"truediv_16_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_16/ReadVariableOp�

truediv_16RealDiv
sub_16:z:0!truediv_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_16Z
Relu_6Relutruediv_16:z:0*
T0*'
_output_shapes
:���������2
Relu_6c
Log1p_6Log1pRelu_6:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_6�
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_19/stack�
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_19/stack_1�
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_19/stack_2�
strided_slice_19StridedSliceinputsstrided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19�
sub_17/ReadVariableOpReadVariableOpsub_7_readvariableop_resource*
_output_shapes
: *
dtype02
sub_17/ReadVariableOp�
sub_17Substrided_slice_19:output:0sub_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_17�
truediv_17/ReadVariableOpReadVariableOp"truediv_17_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_17/ReadVariableOp�

truediv_17RealDiv
sub_17:z:0!truediv_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_17Z
Relu_7Relutruediv_17:z:0*
T0*'
_output_shapes
:���������2
Relu_7c
Log1p_7Log1pRelu_7:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_7�
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_20/stack�
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_20/stack_1�
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_20/stack_2�
strided_slice_20StridedSliceinputsstrided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20�
sub_18/ReadVariableOpReadVariableOpsub_7_readvariableop_resource*
_output_shapes
: *
dtype02
sub_18/ReadVariableOp�
sub_18Substrided_slice_20:output:0sub_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_18�
truediv_18/ReadVariableOpReadVariableOp"truediv_17_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_18/ReadVariableOp�

truediv_18RealDiv
sub_18:z:0!truediv_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_18Z
Relu_8Relutruediv_18:z:0*
T0*'
_output_shapes
:���������2
Relu_8c
Log1p_8Log1pRelu_8:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_8�
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_21/stack�
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_21/stack_1�
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_21/stack_2�
strided_slice_21StridedSliceinputsstrided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21�
sub_19/ReadVariableOpReadVariableOpsub_9_readvariableop_resource*
_output_shapes
: *
dtype02
sub_19/ReadVariableOp�
sub_19Substrided_slice_21:output:0sub_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sub_19�
truediv_19/ReadVariableOpReadVariableOp"truediv_19_readvariableop_resource*
_output_shapes
: *
dtype02
truediv_19/ReadVariableOp�

truediv_19RealDiv
sub_19:z:0!truediv_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

truediv_19Z
Relu_9Relutruediv_19:z:0*
T0*'
_output_shapes
:���������2
Relu_9c
Log1p_9Log1pRelu_9:activations:0*
T0*'
_output_shapes
:���������2	
Log1p_9�
stack_1Pack	Log1p:y:0Log1p_1:y:0Log1p_2:y:0Log1p_3:y:0Log1p_4:y:0Log1p_5:y:0Log1p_6:y:0Log1p_7:y:0Log1p_8:y:0Log1p_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2	
stack_1�
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_22/stack�
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_22/stack_1�
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_22/stack_2�
strided_slice_22StridedSlicestack_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22w
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2
Reshape_3/shape�
	Reshape_3Reshapestrided_slice_22:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:���������
2
	Reshape_3�
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_23/stack�
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_23/stack_1�
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_23/stack_2�
strided_slice_23StridedSlicestack_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23w
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2
Reshape_4/shape�
	Reshape_4Reshapestrided_slice_23:output:0Reshape_4/shape:output:0*
T0*+
_output_shapes
:���������
2
	Reshape_4i
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat_1/axis�
concat_1ConcatV2Reshape_4:output:0stack_1:output:0Reshape_3:output:0concat_1/axis:output:0*
N*
T0*+
_output_shapes
:���������
2

concat_1{
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2
Reshape_5/shape�
	Reshape_5Reshapeconcat_1:output:0Reshape_5/shape:output:0*
T0*/
_output_shapes
:���������
2
	Reshape_5�
IdentityIdentityReshape_2:output:0^sub/ReadVariableOp^sub_1/ReadVariableOp^sub_10/ReadVariableOp^sub_11/ReadVariableOp^sub_12/ReadVariableOp^sub_13/ReadVariableOp^sub_14/ReadVariableOp^sub_15/ReadVariableOp^sub_16/ReadVariableOp^sub_17/ReadVariableOp^sub_18/ReadVariableOp^sub_19/ReadVariableOp^sub_2/ReadVariableOp^sub_3/ReadVariableOp^sub_4/ReadVariableOp^sub_5/ReadVariableOp^sub_6/ReadVariableOp^sub_7/ReadVariableOp^sub_8/ReadVariableOp^sub_9/ReadVariableOp^truediv/ReadVariableOp^truediv_1/ReadVariableOp^truediv_10/ReadVariableOp^truediv_11/ReadVariableOp^truediv_12/ReadVariableOp^truediv_13/ReadVariableOp^truediv_14/ReadVariableOp^truediv_15/ReadVariableOp^truediv_16/ReadVariableOp^truediv_17/ReadVariableOp^truediv_18/ReadVariableOp^truediv_19/ReadVariableOp^truediv_2/ReadVariableOp^truediv_3/ReadVariableOp^truediv_4/ReadVariableOp^truediv_5/ReadVariableOp^truediv_6/ReadVariableOp^truediv_7/ReadVariableOp^truediv_8/ReadVariableOp^truediv_9/ReadVariableOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1IdentityReshape_5:output:0^sub/ReadVariableOp^sub_1/ReadVariableOp^sub_10/ReadVariableOp^sub_11/ReadVariableOp^sub_12/ReadVariableOp^sub_13/ReadVariableOp^sub_14/ReadVariableOp^sub_15/ReadVariableOp^sub_16/ReadVariableOp^sub_17/ReadVariableOp^sub_18/ReadVariableOp^sub_19/ReadVariableOp^sub_2/ReadVariableOp^sub_3/ReadVariableOp^sub_4/ReadVariableOp^sub_5/ReadVariableOp^sub_6/ReadVariableOp^sub_7/ReadVariableOp^sub_8/ReadVariableOp^sub_9/ReadVariableOp^truediv/ReadVariableOp^truediv_1/ReadVariableOp^truediv_10/ReadVariableOp^truediv_11/ReadVariableOp^truediv_12/ReadVariableOp^truediv_13/ReadVariableOp^truediv_14/ReadVariableOp^truediv_15/ReadVariableOp^truediv_16/ReadVariableOp^truediv_17/ReadVariableOp^truediv_18/ReadVariableOp^truediv_19/ReadVariableOp^truediv_2/ReadVariableOp^truediv_3/ReadVariableOp^truediv_4/ReadVariableOp^truediv_5/ReadVariableOp^truediv_6/ReadVariableOp^truediv_7/ReadVariableOp^truediv_8/ReadVariableOp^truediv_9/ReadVariableOp*
T0*/
_output_shapes
:���������
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:���������:::::::::::::::::::::2(
sub/ReadVariableOpsub/ReadVariableOp2,
sub_1/ReadVariableOpsub_1/ReadVariableOp2.
sub_10/ReadVariableOpsub_10/ReadVariableOp2.
sub_11/ReadVariableOpsub_11/ReadVariableOp2.
sub_12/ReadVariableOpsub_12/ReadVariableOp2.
sub_13/ReadVariableOpsub_13/ReadVariableOp2.
sub_14/ReadVariableOpsub_14/ReadVariableOp2.
sub_15/ReadVariableOpsub_15/ReadVariableOp2.
sub_16/ReadVariableOpsub_16/ReadVariableOp2.
sub_17/ReadVariableOpsub_17/ReadVariableOp2.
sub_18/ReadVariableOpsub_18/ReadVariableOp2.
sub_19/ReadVariableOpsub_19/ReadVariableOp2,
sub_2/ReadVariableOpsub_2/ReadVariableOp2,
sub_3/ReadVariableOpsub_3/ReadVariableOp2,
sub_4/ReadVariableOpsub_4/ReadVariableOp2,
sub_5/ReadVariableOpsub_5/ReadVariableOp2,
sub_6/ReadVariableOpsub_6/ReadVariableOp2,
sub_7/ReadVariableOpsub_7/ReadVariableOp2,
sub_8/ReadVariableOpsub_8/ReadVariableOp2,
sub_9/ReadVariableOpsub_9/ReadVariableOp20
truediv/ReadVariableOptruediv/ReadVariableOp24
truediv_1/ReadVariableOptruediv_1/ReadVariableOp26
truediv_10/ReadVariableOptruediv_10/ReadVariableOp26
truediv_11/ReadVariableOptruediv_11/ReadVariableOp26
truediv_12/ReadVariableOptruediv_12/ReadVariableOp26
truediv_13/ReadVariableOptruediv_13/ReadVariableOp26
truediv_14/ReadVariableOptruediv_14/ReadVariableOp26
truediv_15/ReadVariableOptruediv_15/ReadVariableOp26
truediv_16/ReadVariableOptruediv_16/ReadVariableOp26
truediv_17/ReadVariableOptruediv_17/ReadVariableOp26
truediv_18/ReadVariableOptruediv_18/ReadVariableOp26
truediv_19/ReadVariableOptruediv_19/ReadVariableOp24
truediv_2/ReadVariableOptruediv_2/ReadVariableOp24
truediv_3/ReadVariableOptruediv_3/ReadVariableOp24
truediv_4/ReadVariableOptruediv_4/ReadVariableOp24
truediv_5/ReadVariableOptruediv_5/ReadVariableOp24
truediv_6/ReadVariableOptruediv_6/ReadVariableOp24
truediv_7/ReadVariableOptruediv_7/ReadVariableOp24
truediv_8/ReadVariableOptruediv_8/ReadVariableOp24
truediv_9/ReadVariableOptruediv_9/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
l
&__inference_p_re_lu_layer_call_fn_1606

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_p_re_lu_layer_call_and_return_conditional_losses_15982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :������������������:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_2369

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1290

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�%
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2165

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4823

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_2440

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_custom_activation_layer_call_fn_4623

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

unknown_19
identity

identity_1��StatefulPartitionedCall�
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
unknown_19*!
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:���������
:���������
*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_custom_activation_layer_call_and_return_conditional_losses_18602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*/
_output_shapes
:���������
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*~
_input_shapesm
k:���������:::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_5316

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_1313

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13072
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
D
(__inference_flatten_2_layer_call_fn_5070

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
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_22472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
݌
�*
?__inference_model_layer_call_and_return_conditional_losses_4146

inputs1
-custom_activation_sub_readvariableop_resource5
1custom_activation_truediv_readvariableop_resource3
/custom_activation_sub_1_readvariableop_resource7
3custom_activation_truediv_1_readvariableop_resource3
/custom_activation_sub_3_readvariableop_resource7
3custom_activation_truediv_3_readvariableop_resource3
/custom_activation_sub_4_readvariableop_resource7
3custom_activation_truediv_4_readvariableop_resource3
/custom_activation_sub_6_readvariableop_resource7
3custom_activation_truediv_6_readvariableop_resource3
/custom_activation_sub_7_readvariableop_resource7
3custom_activation_truediv_7_readvariableop_resource3
/custom_activation_sub_9_readvariableop_resource7
3custom_activation_truediv_9_readvariableop_resource8
4custom_activation_truediv_10_readvariableop_resource8
4custom_activation_truediv_11_readvariableop_resource8
4custom_activation_truediv_13_readvariableop_resource8
4custom_activation_truediv_14_readvariableop_resource8
4custom_activation_truediv_16_readvariableop_resource8
4custom_activation_truediv_17_readvariableop_resource8
4custom_activation_truediv_19_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource#
p_re_lu_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�$custom_activation/sub/ReadVariableOp�&custom_activation/sub_1/ReadVariableOp�'custom_activation/sub_10/ReadVariableOp�'custom_activation/sub_11/ReadVariableOp�'custom_activation/sub_12/ReadVariableOp�'custom_activation/sub_13/ReadVariableOp�'custom_activation/sub_14/ReadVariableOp�'custom_activation/sub_15/ReadVariableOp�'custom_activation/sub_16/ReadVariableOp�'custom_activation/sub_17/ReadVariableOp�'custom_activation/sub_18/ReadVariableOp�'custom_activation/sub_19/ReadVariableOp�&custom_activation/sub_2/ReadVariableOp�&custom_activation/sub_3/ReadVariableOp�&custom_activation/sub_4/ReadVariableOp�&custom_activation/sub_5/ReadVariableOp�&custom_activation/sub_6/ReadVariableOp�&custom_activation/sub_7/ReadVariableOp�&custom_activation/sub_8/ReadVariableOp�&custom_activation/sub_9/ReadVariableOp�(custom_activation/truediv/ReadVariableOp�*custom_activation/truediv_1/ReadVariableOp�+custom_activation/truediv_10/ReadVariableOp�+custom_activation/truediv_11/ReadVariableOp�+custom_activation/truediv_12/ReadVariableOp�+custom_activation/truediv_13/ReadVariableOp�+custom_activation/truediv_14/ReadVariableOp�+custom_activation/truediv_15/ReadVariableOp�+custom_activation/truediv_16/ReadVariableOp�+custom_activation/truediv_17/ReadVariableOp�+custom_activation/truediv_18/ReadVariableOp�+custom_activation/truediv_19/ReadVariableOp�*custom_activation/truediv_2/ReadVariableOp�*custom_activation/truediv_3/ReadVariableOp�*custom_activation/truediv_4/ReadVariableOp�*custom_activation/truediv_5/ReadVariableOp�*custom_activation/truediv_6/ReadVariableOp�*custom_activation/truediv_7/ReadVariableOp�*custom_activation/truediv_8/ReadVariableOp�*custom_activation/truediv_9/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�p_re_lu/ReadVariableOp�
%custom_activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2'
%custom_activation/strided_slice/stack�
'custom_activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice/stack_1�
'custom_activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2)
'custom_activation/strided_slice/stack_2�
custom_activation/strided_sliceStridedSliceinputs.custom_activation/strided_slice/stack:output:00custom_activation/strided_slice/stack_1:output:00custom_activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2!
custom_activation/strided_slice�
$custom_activation/sub/ReadVariableOpReadVariableOp-custom_activation_sub_readvariableop_resource*
_output_shapes
: *
dtype02&
$custom_activation/sub/ReadVariableOp�
custom_activation/subSub(custom_activation/strided_slice:output:0,custom_activation/sub/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub�
(custom_activation/truediv/ReadVariableOpReadVariableOp1custom_activation_truediv_readvariableop_resource*
_output_shapes
: *
dtype02*
(custom_activation/truediv/ReadVariableOp�
custom_activation/truedivRealDivcustom_activation/sub:z:00custom_activation/truediv/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv�
custom_activation/TanhTanhcustom_activation/truediv:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh�
'custom_activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_1/stack�
)custom_activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_1/stack_1�
)custom_activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_1/stack_2�
!custom_activation/strided_slice_1StridedSliceinputs0custom_activation/strided_slice_1/stack:output:02custom_activation/strided_slice_1/stack_1:output:02custom_activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_1�
&custom_activation/sub_1/ReadVariableOpReadVariableOp/custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_1/ReadVariableOp�
custom_activation/sub_1Sub*custom_activation/strided_slice_1:output:0.custom_activation/sub_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_1�
*custom_activation/truediv_1/ReadVariableOpReadVariableOp3custom_activation_truediv_1_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_1/ReadVariableOp�
custom_activation/truediv_1RealDivcustom_activation/sub_1:z:02custom_activation/truediv_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_1�
custom_activation/Tanh_1Tanhcustom_activation/truediv_1:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_1�
'custom_activation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_2/stack�
)custom_activation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_2/stack_1�
)custom_activation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_2/stack_2�
!custom_activation/strided_slice_2StridedSliceinputs0custom_activation/strided_slice_2/stack:output:02custom_activation/strided_slice_2/stack_1:output:02custom_activation/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_2�
&custom_activation/sub_2/ReadVariableOpReadVariableOp/custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_2/ReadVariableOp�
custom_activation/sub_2Sub*custom_activation/strided_slice_2:output:0.custom_activation/sub_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_2�
*custom_activation/truediv_2/ReadVariableOpReadVariableOp3custom_activation_truediv_1_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_2/ReadVariableOp�
custom_activation/truediv_2RealDivcustom_activation/sub_2:z:02custom_activation/truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_2�
custom_activation/Tanh_2Tanhcustom_activation/truediv_2:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_2�
'custom_activation/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2)
'custom_activation/strided_slice_3/stack�
)custom_activation/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_3/stack_1�
)custom_activation/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_3/stack_2�
!custom_activation/strided_slice_3StridedSliceinputs0custom_activation/strided_slice_3/stack:output:02custom_activation/strided_slice_3/stack_1:output:02custom_activation/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_3�
&custom_activation/sub_3/ReadVariableOpReadVariableOp/custom_activation_sub_3_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_3/ReadVariableOp�
custom_activation/sub_3Sub*custom_activation/strided_slice_3:output:0.custom_activation/sub_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_3�
*custom_activation/truediv_3/ReadVariableOpReadVariableOp3custom_activation_truediv_3_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_3/ReadVariableOp�
custom_activation/truediv_3RealDivcustom_activation/sub_3:z:02custom_activation/truediv_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_3�
custom_activation/Tanh_3Tanhcustom_activation/truediv_3:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_3�
'custom_activation/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_4/stack�
)custom_activation/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_4/stack_1�
)custom_activation/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_4/stack_2�
!custom_activation/strided_slice_4StridedSliceinputs0custom_activation/strided_slice_4/stack:output:02custom_activation/strided_slice_4/stack_1:output:02custom_activation/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_4�
&custom_activation/sub_4/ReadVariableOpReadVariableOp/custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_4/ReadVariableOp�
custom_activation/sub_4Sub*custom_activation/strided_slice_4:output:0.custom_activation/sub_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_4�
*custom_activation/truediv_4/ReadVariableOpReadVariableOp3custom_activation_truediv_4_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_4/ReadVariableOp�
custom_activation/truediv_4RealDivcustom_activation/sub_4:z:02custom_activation/truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_4�
custom_activation/Tanh_4Tanhcustom_activation/truediv_4:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_4�
'custom_activation/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_5/stack�
)custom_activation/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_5/stack_1�
)custom_activation/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_5/stack_2�
!custom_activation/strided_slice_5StridedSliceinputs0custom_activation/strided_slice_5/stack:output:02custom_activation/strided_slice_5/stack_1:output:02custom_activation/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_5�
&custom_activation/sub_5/ReadVariableOpReadVariableOp/custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_5/ReadVariableOp�
custom_activation/sub_5Sub*custom_activation/strided_slice_5:output:0.custom_activation/sub_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_5�
*custom_activation/truediv_5/ReadVariableOpReadVariableOp3custom_activation_truediv_4_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_5/ReadVariableOp�
custom_activation/truediv_5RealDivcustom_activation/sub_5:z:02custom_activation/truediv_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_5�
custom_activation/Tanh_5Tanhcustom_activation/truediv_5:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_5�
'custom_activation/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2)
'custom_activation/strided_slice_6/stack�
)custom_activation/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_6/stack_1�
)custom_activation/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_6/stack_2�
!custom_activation/strided_slice_6StridedSliceinputs0custom_activation/strided_slice_6/stack:output:02custom_activation/strided_slice_6/stack_1:output:02custom_activation/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_6�
&custom_activation/sub_6/ReadVariableOpReadVariableOp/custom_activation_sub_6_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_6/ReadVariableOp�
custom_activation/sub_6Sub*custom_activation/strided_slice_6:output:0.custom_activation/sub_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_6�
*custom_activation/truediv_6/ReadVariableOpReadVariableOp3custom_activation_truediv_6_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_6/ReadVariableOp�
custom_activation/truediv_6RealDivcustom_activation/sub_6:z:02custom_activation/truediv_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_6�
custom_activation/Tanh_6Tanhcustom_activation/truediv_6:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_6�
'custom_activation/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_7/stack�
)custom_activation/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_7/stack_1�
)custom_activation/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_7/stack_2�
!custom_activation/strided_slice_7StridedSliceinputs0custom_activation/strided_slice_7/stack:output:02custom_activation/strided_slice_7/stack_1:output:02custom_activation/strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_7�
&custom_activation/sub_7/ReadVariableOpReadVariableOp/custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_7/ReadVariableOp�
custom_activation/sub_7Sub*custom_activation/strided_slice_7:output:0.custom_activation/sub_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_7�
*custom_activation/truediv_7/ReadVariableOpReadVariableOp3custom_activation_truediv_7_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_7/ReadVariableOp�
custom_activation/truediv_7RealDivcustom_activation/sub_7:z:02custom_activation/truediv_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_7�
custom_activation/Tanh_7Tanhcustom_activation/truediv_7:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_7�
'custom_activation/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'custom_activation/strided_slice_8/stack�
)custom_activation/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_8/stack_1�
)custom_activation/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_8/stack_2�
!custom_activation/strided_slice_8StridedSliceinputs0custom_activation/strided_slice_8/stack:output:02custom_activation/strided_slice_8/stack_1:output:02custom_activation/strided_slice_8/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_8�
&custom_activation/sub_8/ReadVariableOpReadVariableOp/custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_8/ReadVariableOp�
custom_activation/sub_8Sub*custom_activation/strided_slice_8:output:0.custom_activation/sub_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_8�
*custom_activation/truediv_8/ReadVariableOpReadVariableOp3custom_activation_truediv_7_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_8/ReadVariableOp�
custom_activation/truediv_8RealDivcustom_activation/sub_8:z:02custom_activation/truediv_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_8�
custom_activation/Tanh_8Tanhcustom_activation/truediv_8:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_8�
'custom_activation/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2)
'custom_activation/strided_slice_9/stack�
)custom_activation/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)custom_activation/strided_slice_9/stack_1�
)custom_activation/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)custom_activation/strided_slice_9/stack_2�
!custom_activation/strided_slice_9StridedSliceinputs0custom_activation/strided_slice_9/stack:output:02custom_activation/strided_slice_9/stack_1:output:02custom_activation/strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2#
!custom_activation/strided_slice_9�
&custom_activation/sub_9/ReadVariableOpReadVariableOp/custom_activation_sub_9_readvariableop_resource*
_output_shapes
: *
dtype02(
&custom_activation/sub_9/ReadVariableOp�
custom_activation/sub_9Sub*custom_activation/strided_slice_9:output:0.custom_activation/sub_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_9�
*custom_activation/truediv_9/ReadVariableOpReadVariableOp3custom_activation_truediv_9_readvariableop_resource*
_output_shapes
: *
dtype02,
*custom_activation/truediv_9/ReadVariableOp�
custom_activation/truediv_9RealDivcustom_activation/sub_9:z:02custom_activation/truediv_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_9�
custom_activation/Tanh_9Tanhcustom_activation/truediv_9:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Tanh_9�
custom_activation/stackPackcustom_activation/Tanh:y:0custom_activation/Tanh_1:y:0custom_activation/Tanh_2:y:0custom_activation/Tanh_3:y:0custom_activation/Tanh_4:y:0custom_activation/Tanh_5:y:0custom_activation/Tanh_6:y:0custom_activation/Tanh_7:y:0custom_activation/Tanh_8:y:0custom_activation/Tanh_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2
custom_activation/stack�
(custom_activation/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_10/stack�
*custom_activation/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_10/stack_1�
*custom_activation/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_10/stack_2�
"custom_activation/strided_slice_10StridedSlice custom_activation/stack:output:01custom_activation/strided_slice_10/stack:output:03custom_activation/strided_slice_10/stack_1:output:03custom_activation/strided_slice_10/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_10�
custom_activation/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2!
custom_activation/Reshape/shape�
custom_activation/ReshapeReshape+custom_activation/strided_slice_10:output:0(custom_activation/Reshape/shape:output:0*
T0*+
_output_shapes
:���������
2
custom_activation/Reshape�
(custom_activation/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_11/stack�
*custom_activation/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_11/stack_1�
*custom_activation/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_11/stack_2�
"custom_activation/strided_slice_11StridedSlice custom_activation/stack:output:01custom_activation/strided_slice_11/stack:output:03custom_activation/strided_slice_11/stack_1:output:03custom_activation/strided_slice_11/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_11�
!custom_activation/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2#
!custom_activation/Reshape_1/shape�
custom_activation/Reshape_1Reshape+custom_activation/strided_slice_11:output:0*custom_activation/Reshape_1/shape:output:0*
T0*+
_output_shapes
:���������
2
custom_activation/Reshape_1�
custom_activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
custom_activation/concat/axis�
custom_activation/concatConcatV2$custom_activation/Reshape_1:output:0 custom_activation/stack:output:0"custom_activation/Reshape:output:0&custom_activation/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
2
custom_activation/concat�
!custom_activation/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2#
!custom_activation/Reshape_2/shape�
custom_activation/Reshape_2Reshape!custom_activation/concat:output:0*custom_activation/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������
2
custom_activation/Reshape_2�
(custom_activation/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_12/stack�
*custom_activation/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_12/stack_1�
*custom_activation/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_12/stack_2�
"custom_activation/strided_slice_12StridedSliceinputs1custom_activation/strided_slice_12/stack:output:03custom_activation/strided_slice_12/stack_1:output:03custom_activation/strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_12�
'custom_activation/sub_10/ReadVariableOpReadVariableOp-custom_activation_sub_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_10/ReadVariableOp�
custom_activation/sub_10Sub+custom_activation/strided_slice_12:output:0/custom_activation/sub_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_10�
+custom_activation/truediv_10/ReadVariableOpReadVariableOp4custom_activation_truediv_10_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_10/ReadVariableOp�
custom_activation/truediv_10RealDivcustom_activation/sub_10:z:03custom_activation/truediv_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_10�
custom_activation/ReluRelu custom_activation/truediv_10:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu�
custom_activation/Log1pLog1p$custom_activation/Relu:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p�
(custom_activation/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_13/stack�
*custom_activation/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_13/stack_1�
*custom_activation/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_13/stack_2�
"custom_activation/strided_slice_13StridedSliceinputs1custom_activation/strided_slice_13/stack:output:03custom_activation/strided_slice_13/stack_1:output:03custom_activation/strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_13�
'custom_activation/sub_11/ReadVariableOpReadVariableOp/custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_11/ReadVariableOp�
custom_activation/sub_11Sub+custom_activation/strided_slice_13:output:0/custom_activation/sub_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_11�
+custom_activation/truediv_11/ReadVariableOpReadVariableOp4custom_activation_truediv_11_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_11/ReadVariableOp�
custom_activation/truediv_11RealDivcustom_activation/sub_11:z:03custom_activation/truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_11�
custom_activation/Relu_1Relu custom_activation/truediv_11:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_1�
custom_activation/Log1p_1Log1p&custom_activation/Relu_1:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_1�
(custom_activation/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_14/stack�
*custom_activation/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_14/stack_1�
*custom_activation/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_14/stack_2�
"custom_activation/strided_slice_14StridedSliceinputs1custom_activation/strided_slice_14/stack:output:03custom_activation/strided_slice_14/stack_1:output:03custom_activation/strided_slice_14/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_14�
'custom_activation/sub_12/ReadVariableOpReadVariableOp/custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_12/ReadVariableOp�
custom_activation/sub_12Sub+custom_activation/strided_slice_14:output:0/custom_activation/sub_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_12�
+custom_activation/truediv_12/ReadVariableOpReadVariableOp4custom_activation_truediv_11_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_12/ReadVariableOp�
custom_activation/truediv_12RealDivcustom_activation/sub_12:z:03custom_activation/truediv_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_12�
custom_activation/Relu_2Relu custom_activation/truediv_12:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_2�
custom_activation/Log1p_2Log1p&custom_activation/Relu_2:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_2�
(custom_activation/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_15/stack�
*custom_activation/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_15/stack_1�
*custom_activation/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_15/stack_2�
"custom_activation/strided_slice_15StridedSliceinputs1custom_activation/strided_slice_15/stack:output:03custom_activation/strided_slice_15/stack_1:output:03custom_activation/strided_slice_15/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_15�
'custom_activation/sub_13/ReadVariableOpReadVariableOp/custom_activation_sub_3_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_13/ReadVariableOp�
custom_activation/sub_13Sub+custom_activation/strided_slice_15:output:0/custom_activation/sub_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_13�
+custom_activation/truediv_13/ReadVariableOpReadVariableOp4custom_activation_truediv_13_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_13/ReadVariableOp�
custom_activation/truediv_13RealDivcustom_activation/sub_13:z:03custom_activation/truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_13�
custom_activation/Relu_3Relu custom_activation/truediv_13:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_3�
custom_activation/Log1p_3Log1p&custom_activation/Relu_3:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_3�
(custom_activation/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_16/stack�
*custom_activation/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_16/stack_1�
*custom_activation/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_16/stack_2�
"custom_activation/strided_slice_16StridedSliceinputs1custom_activation/strided_slice_16/stack:output:03custom_activation/strided_slice_16/stack_1:output:03custom_activation/strided_slice_16/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_16�
'custom_activation/sub_14/ReadVariableOpReadVariableOp/custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_14/ReadVariableOp�
custom_activation/sub_14Sub+custom_activation/strided_slice_16:output:0/custom_activation/sub_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_14�
+custom_activation/truediv_14/ReadVariableOpReadVariableOp4custom_activation_truediv_14_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_14/ReadVariableOp�
custom_activation/truediv_14RealDivcustom_activation/sub_14:z:03custom_activation/truediv_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_14�
custom_activation/Relu_4Relu custom_activation/truediv_14:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_4�
custom_activation/Log1p_4Log1p&custom_activation/Relu_4:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_4�
(custom_activation/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_17/stack�
*custom_activation/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_17/stack_1�
*custom_activation/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_17/stack_2�
"custom_activation/strided_slice_17StridedSliceinputs1custom_activation/strided_slice_17/stack:output:03custom_activation/strided_slice_17/stack_1:output:03custom_activation/strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_17�
'custom_activation/sub_15/ReadVariableOpReadVariableOp/custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_15/ReadVariableOp�
custom_activation/sub_15Sub+custom_activation/strided_slice_17:output:0/custom_activation/sub_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_15�
+custom_activation/truediv_15/ReadVariableOpReadVariableOp4custom_activation_truediv_14_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_15/ReadVariableOp�
custom_activation/truediv_15RealDivcustom_activation/sub_15:z:03custom_activation/truediv_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_15�
custom_activation/Relu_5Relu custom_activation/truediv_15:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_5�
custom_activation/Log1p_5Log1p&custom_activation/Relu_5:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_5�
(custom_activation/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_18/stack�
*custom_activation/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_18/stack_1�
*custom_activation/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_18/stack_2�
"custom_activation/strided_slice_18StridedSliceinputs1custom_activation/strided_slice_18/stack:output:03custom_activation/strided_slice_18/stack_1:output:03custom_activation/strided_slice_18/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_18�
'custom_activation/sub_16/ReadVariableOpReadVariableOp/custom_activation_sub_6_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_16/ReadVariableOp�
custom_activation/sub_16Sub+custom_activation/strided_slice_18:output:0/custom_activation/sub_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_16�
+custom_activation/truediv_16/ReadVariableOpReadVariableOp4custom_activation_truediv_16_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_16/ReadVariableOp�
custom_activation/truediv_16RealDivcustom_activation/sub_16:z:03custom_activation/truediv_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_16�
custom_activation/Relu_6Relu custom_activation/truediv_16:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_6�
custom_activation/Log1p_6Log1p&custom_activation/Relu_6:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_6�
(custom_activation/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_19/stack�
*custom_activation/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_19/stack_1�
*custom_activation/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_19/stack_2�
"custom_activation/strided_slice_19StridedSliceinputs1custom_activation/strided_slice_19/stack:output:03custom_activation/strided_slice_19/stack_1:output:03custom_activation/strided_slice_19/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_19�
'custom_activation/sub_17/ReadVariableOpReadVariableOp/custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_17/ReadVariableOp�
custom_activation/sub_17Sub+custom_activation/strided_slice_19:output:0/custom_activation/sub_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_17�
+custom_activation/truediv_17/ReadVariableOpReadVariableOp4custom_activation_truediv_17_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_17/ReadVariableOp�
custom_activation/truediv_17RealDivcustom_activation/sub_17:z:03custom_activation/truediv_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_17�
custom_activation/Relu_7Relu custom_activation/truediv_17:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_7�
custom_activation/Log1p_7Log1p&custom_activation/Relu_7:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_7�
(custom_activation/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_20/stack�
*custom_activation/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_20/stack_1�
*custom_activation/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_20/stack_2�
"custom_activation/strided_slice_20StridedSliceinputs1custom_activation/strided_slice_20/stack:output:03custom_activation/strided_slice_20/stack_1:output:03custom_activation/strided_slice_20/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_20�
'custom_activation/sub_18/ReadVariableOpReadVariableOp/custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_18/ReadVariableOp�
custom_activation/sub_18Sub+custom_activation/strided_slice_20:output:0/custom_activation/sub_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_18�
+custom_activation/truediv_18/ReadVariableOpReadVariableOp4custom_activation_truediv_17_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_18/ReadVariableOp�
custom_activation/truediv_18RealDivcustom_activation/sub_18:z:03custom_activation/truediv_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_18�
custom_activation/Relu_8Relu custom_activation/truediv_18:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_8�
custom_activation/Log1p_8Log1p&custom_activation/Relu_8:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_8�
(custom_activation/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_21/stack�
*custom_activation/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_21/stack_1�
*custom_activation/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_21/stack_2�
"custom_activation/strided_slice_21StridedSliceinputs1custom_activation/strided_slice_21/stack:output:03custom_activation/strided_slice_21/stack_1:output:03custom_activation/strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_21�
'custom_activation/sub_19/ReadVariableOpReadVariableOp/custom_activation_sub_9_readvariableop_resource*
_output_shapes
: *
dtype02)
'custom_activation/sub_19/ReadVariableOp�
custom_activation/sub_19Sub+custom_activation/strided_slice_21:output:0/custom_activation/sub_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/sub_19�
+custom_activation/truediv_19/ReadVariableOpReadVariableOp4custom_activation_truediv_19_readvariableop_resource*
_output_shapes
: *
dtype02-
+custom_activation/truediv_19/ReadVariableOp�
custom_activation/truediv_19RealDivcustom_activation/sub_19:z:03custom_activation/truediv_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
custom_activation/truediv_19�
custom_activation/Relu_9Relu custom_activation/truediv_19:z:0*
T0*'
_output_shapes
:���������2
custom_activation/Relu_9�
custom_activation/Log1p_9Log1p&custom_activation/Relu_9:activations:0*
T0*'
_output_shapes
:���������2
custom_activation/Log1p_9�
custom_activation/stack_1Packcustom_activation/Log1p:y:0custom_activation/Log1p_1:y:0custom_activation/Log1p_2:y:0custom_activation/Log1p_3:y:0custom_activation/Log1p_4:y:0custom_activation/Log1p_5:y:0custom_activation/Log1p_6:y:0custom_activation/Log1p_7:y:0custom_activation/Log1p_8:y:0custom_activation/Log1p_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2
custom_activation/stack_1�
(custom_activation/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2*
(custom_activation/strided_slice_22/stack�
*custom_activation/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_22/stack_1�
*custom_activation/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_22/stack_2�
"custom_activation/strided_slice_22StridedSlice"custom_activation/stack_1:output:01custom_activation/strided_slice_22/stack:output:03custom_activation/strided_slice_22/stack_1:output:03custom_activation/strided_slice_22/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_22�
!custom_activation/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2#
!custom_activation/Reshape_3/shape�
custom_activation/Reshape_3Reshape+custom_activation/strided_slice_22:output:0*custom_activation/Reshape_3/shape:output:0*
T0*+
_output_shapes
:���������
2
custom_activation/Reshape_3�
(custom_activation/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2*
(custom_activation/strided_slice_23/stack�
*custom_activation/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2,
*custom_activation/strided_slice_23/stack_1�
*custom_activation/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2,
*custom_activation/strided_slice_23/stack_2�
"custom_activation/strided_slice_23StridedSlice"custom_activation/stack_1:output:01custom_activation/strided_slice_23/stack:output:03custom_activation/strided_slice_23/stack_1:output:03custom_activation/strided_slice_23/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2$
"custom_activation/strided_slice_23�
!custom_activation/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2#
!custom_activation/Reshape_4/shape�
custom_activation/Reshape_4Reshape+custom_activation/strided_slice_23:output:0*custom_activation/Reshape_4/shape:output:0*
T0*+
_output_shapes
:���������
2
custom_activation/Reshape_4�
custom_activation/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
custom_activation/concat_1/axis�
custom_activation/concat_1ConcatV2$custom_activation/Reshape_4:output:0"custom_activation/stack_1:output:0$custom_activation/Reshape_3:output:0(custom_activation/concat_1/axis:output:0*
N*
T0*+
_output_shapes
:���������
2
custom_activation/concat_1�
!custom_activation/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2#
!custom_activation/Reshape_5/shape�
custom_activation/Reshape_5Reshape#custom_activation/concat_1:output:0*custom_activation/Reshape_5/shape:output:0*
T0*/
_output_shapes
:���������
2
custom_activation/Reshape_5�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2D$custom_activation/Reshape_5:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2d_1/BiasAdd�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2D$custom_activation/Reshape_2:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
conv2d/BiasAdd�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3�
tf.math.tanh_1/TanhTanh*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
tf.math.tanh_1/Tanh�
tf.nn.relu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
tf.nn.relu_1/Relu�
tf.math.tanh/TanhTanh(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
tf.math.tanh/Tanh�
tf.nn.relu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
tf.nn.relu/Relu�
average_pooling2d_3/AvgPoolAvgPooltf.math.tanh_1/Tanh:y:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_3/AvgPool�
max_pooling2d_3/MaxPoolMaxPooltf.math.tanh_1/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool�
average_pooling2d_2/AvgPoolAvgPooltf.nn.relu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_2/AvgPool�
max_pooling2d_2/MaxPoolMaxPooltf.nn.relu_1/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool�
average_pooling2d_1/AvgPoolAvgPooltf.math.tanh/Tanh:y:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool�
max_pooling2d_1/MaxPoolMaxPooltf.math.tanh/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool�
average_pooling2d/AvgPoolAvgPooltf.nn.relu/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool�
max_pooling2d/MaxPoolMaxPooltf.nn.relu/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_1/Const�
flatten_1/ReshapeReshape"average_pooling2d/AvgPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_2/Const�
flatten_2/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_2/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_3/Const�
flatten_3/ReshapeReshape$average_pooling2d_1/AvgPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_3/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_4/Const�
flatten_4/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_4/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_5/Const�
flatten_5/ReshapeReshape$average_pooling2d_2/AvgPool:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_5/Reshapes
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_6/Const�
flatten_6/ReshapeReshape max_pooling2d_3/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_6/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten_7/Const�
flatten_7/ReshapeReshape$average_pooling2d_3/AvgPool:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_7/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis�
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0flatten_5/Reshape:output:0flatten_6/Reshape:output:0flatten_7/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate/concat�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAdd�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp�
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_2/batchnorm/add/y�
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/add�
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/Rsqrt�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/mul�
%batch_normalization_2/batchnorm/mul_1Muldense/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/mul_1�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_2/batchnorm/mul_2�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_2/batchnorm/sub�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_2/batchnorm/add_1�
p_re_lu/ReluRelu)batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
p_re_lu/Relu�
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*
_output_shapes	
:�*
dtype02
p_re_lu/ReadVariableOpg
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
p_re_lu/Neg�
p_re_lu/Neg_1Neg)batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
p_re_lu/Neg_1n
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*(
_output_shapes
:����������2
p_re_lu/Relu_1�
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*(
_output_shapes
:����������2
p_re_lu/mul�
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*(
_output_shapes
:����������2
p_re_lu/addt
dropout/IdentityIdentityp_re_lu/add:z:0*
T0*(
_output_shapes
:����������2
dropout/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Softmax�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentitydense_1/Softmax:softmax:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp%^custom_activation/sub/ReadVariableOp'^custom_activation/sub_1/ReadVariableOp(^custom_activation/sub_10/ReadVariableOp(^custom_activation/sub_11/ReadVariableOp(^custom_activation/sub_12/ReadVariableOp(^custom_activation/sub_13/ReadVariableOp(^custom_activation/sub_14/ReadVariableOp(^custom_activation/sub_15/ReadVariableOp(^custom_activation/sub_16/ReadVariableOp(^custom_activation/sub_17/ReadVariableOp(^custom_activation/sub_18/ReadVariableOp(^custom_activation/sub_19/ReadVariableOp'^custom_activation/sub_2/ReadVariableOp'^custom_activation/sub_3/ReadVariableOp'^custom_activation/sub_4/ReadVariableOp'^custom_activation/sub_5/ReadVariableOp'^custom_activation/sub_6/ReadVariableOp'^custom_activation/sub_7/ReadVariableOp'^custom_activation/sub_8/ReadVariableOp'^custom_activation/sub_9/ReadVariableOp)^custom_activation/truediv/ReadVariableOp+^custom_activation/truediv_1/ReadVariableOp,^custom_activation/truediv_10/ReadVariableOp,^custom_activation/truediv_11/ReadVariableOp,^custom_activation/truediv_12/ReadVariableOp,^custom_activation/truediv_13/ReadVariableOp,^custom_activation/truediv_14/ReadVariableOp,^custom_activation/truediv_15/ReadVariableOp,^custom_activation/truediv_16/ReadVariableOp,^custom_activation/truediv_17/ReadVariableOp,^custom_activation/truediv_18/ReadVariableOp,^custom_activation/truediv_19/ReadVariableOp+^custom_activation/truediv_2/ReadVariableOp+^custom_activation/truediv_3/ReadVariableOp+^custom_activation/truediv_4/ReadVariableOp+^custom_activation/truediv_5/ReadVariableOp+^custom_activation/truediv_6/ReadVariableOp+^custom_activation/truediv_7/ReadVariableOp+^custom_activation/truediv_8/ReadVariableOp+^custom_activation/truediv_9/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^p_re_lu/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2L
$custom_activation/sub/ReadVariableOp$custom_activation/sub/ReadVariableOp2P
&custom_activation/sub_1/ReadVariableOp&custom_activation/sub_1/ReadVariableOp2R
'custom_activation/sub_10/ReadVariableOp'custom_activation/sub_10/ReadVariableOp2R
'custom_activation/sub_11/ReadVariableOp'custom_activation/sub_11/ReadVariableOp2R
'custom_activation/sub_12/ReadVariableOp'custom_activation/sub_12/ReadVariableOp2R
'custom_activation/sub_13/ReadVariableOp'custom_activation/sub_13/ReadVariableOp2R
'custom_activation/sub_14/ReadVariableOp'custom_activation/sub_14/ReadVariableOp2R
'custom_activation/sub_15/ReadVariableOp'custom_activation/sub_15/ReadVariableOp2R
'custom_activation/sub_16/ReadVariableOp'custom_activation/sub_16/ReadVariableOp2R
'custom_activation/sub_17/ReadVariableOp'custom_activation/sub_17/ReadVariableOp2R
'custom_activation/sub_18/ReadVariableOp'custom_activation/sub_18/ReadVariableOp2R
'custom_activation/sub_19/ReadVariableOp'custom_activation/sub_19/ReadVariableOp2P
&custom_activation/sub_2/ReadVariableOp&custom_activation/sub_2/ReadVariableOp2P
&custom_activation/sub_3/ReadVariableOp&custom_activation/sub_3/ReadVariableOp2P
&custom_activation/sub_4/ReadVariableOp&custom_activation/sub_4/ReadVariableOp2P
&custom_activation/sub_5/ReadVariableOp&custom_activation/sub_5/ReadVariableOp2P
&custom_activation/sub_6/ReadVariableOp&custom_activation/sub_6/ReadVariableOp2P
&custom_activation/sub_7/ReadVariableOp&custom_activation/sub_7/ReadVariableOp2P
&custom_activation/sub_8/ReadVariableOp&custom_activation/sub_8/ReadVariableOp2P
&custom_activation/sub_9/ReadVariableOp&custom_activation/sub_9/ReadVariableOp2T
(custom_activation/truediv/ReadVariableOp(custom_activation/truediv/ReadVariableOp2X
*custom_activation/truediv_1/ReadVariableOp*custom_activation/truediv_1/ReadVariableOp2Z
+custom_activation/truediv_10/ReadVariableOp+custom_activation/truediv_10/ReadVariableOp2Z
+custom_activation/truediv_11/ReadVariableOp+custom_activation/truediv_11/ReadVariableOp2Z
+custom_activation/truediv_12/ReadVariableOp+custom_activation/truediv_12/ReadVariableOp2Z
+custom_activation/truediv_13/ReadVariableOp+custom_activation/truediv_13/ReadVariableOp2Z
+custom_activation/truediv_14/ReadVariableOp+custom_activation/truediv_14/ReadVariableOp2Z
+custom_activation/truediv_15/ReadVariableOp+custom_activation/truediv_15/ReadVariableOp2Z
+custom_activation/truediv_16/ReadVariableOp+custom_activation/truediv_16/ReadVariableOp2Z
+custom_activation/truediv_17/ReadVariableOp+custom_activation/truediv_17/ReadVariableOp2Z
+custom_activation/truediv_18/ReadVariableOp+custom_activation/truediv_18/ReadVariableOp2Z
+custom_activation/truediv_19/ReadVariableOp+custom_activation/truediv_19/ReadVariableOp2X
*custom_activation/truediv_2/ReadVariableOp*custom_activation/truediv_2/ReadVariableOp2X
*custom_activation/truediv_3/ReadVariableOp*custom_activation/truediv_3/ReadVariableOp2X
*custom_activation/truediv_4/ReadVariableOp*custom_activation/truediv_4/ReadVariableOp2X
*custom_activation/truediv_5/ReadVariableOp*custom_activation/truediv_5/ReadVariableOp2X
*custom_activation/truediv_6/ReadVariableOp*custom_activation/truediv_6/ReadVariableOp2X
*custom_activation/truediv_7/ReadVariableOp*custom_activation/truediv_7/ReadVariableOp2X
*custom_activation/truediv_8/ReadVariableOp*custom_activation/truediv_8/ReadVariableOp2X
*custom_activation/truediv_9/ReadVariableOp*custom_activation/truediv_9/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp20
p_re_lu/ReadVariableOpp_re_lu/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_flatten_7_layer_call_and_return_conditional_losses_5120

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2068

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_5172

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1331

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4652

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_5337

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_3_layer_call_fn_1385

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_13792
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�G
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5241

inputs
assignmovingavg_5204
assignmovingavg_1_5210)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/5204*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_5204*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/5204*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/5204*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_5204AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/5204*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/5210*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_5210*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/5210*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/5210*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_5210AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/5210*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_2435

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
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
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_average_pooling2d_2_layer_call_fn_1373

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_13672
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
2__inference_batch_normalization_layer_call_fn_4836

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_21352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
|
'__inference_conv2d_1_layer_call_fn_4661

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_19652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
��
�+
__inference__wrapped_model_997
input_17
3model_custom_activation_sub_readvariableop_resource;
7model_custom_activation_truediv_readvariableop_resource9
5model_custom_activation_sub_1_readvariableop_resource=
9model_custom_activation_truediv_1_readvariableop_resource9
5model_custom_activation_sub_3_readvariableop_resource=
9model_custom_activation_truediv_3_readvariableop_resource9
5model_custom_activation_sub_4_readvariableop_resource=
9model_custom_activation_truediv_4_readvariableop_resource9
5model_custom_activation_sub_6_readvariableop_resource=
9model_custom_activation_truediv_6_readvariableop_resource9
5model_custom_activation_sub_7_readvariableop_resource=
9model_custom_activation_truediv_7_readvariableop_resource9
5model_custom_activation_sub_9_readvariableop_resource=
9model_custom_activation_truediv_9_readvariableop_resource>
:model_custom_activation_truediv_10_readvariableop_resource>
:model_custom_activation_truediv_11_readvariableop_resource>
:model_custom_activation_truediv_13_readvariableop_resource>
:model_custom_activation_truediv_14_readvariableop_resource>
:model_custom_activation_truediv_16_readvariableop_resource>
:model_custom_activation_truediv_17_readvariableop_resource>
:model_custom_activation_truediv_19_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource7
3model_batch_normalization_1_readvariableop_resource9
5model_batch_normalization_1_readvariableop_1_resourceH
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource5
1model_batch_normalization_readvariableop_resource7
3model_batch_normalization_readvariableop_1_resourceF
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resourceH
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resourceA
=model_batch_normalization_2_batchnorm_readvariableop_resourceE
Amodel_batch_normalization_2_batchnorm_mul_readvariableop_resourceC
?model_batch_normalization_2_batchnorm_readvariableop_1_resourceC
?model_batch_normalization_2_batchnorm_readvariableop_2_resource)
%model_p_re_lu_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource
identity��9model/batch_normalization/FusedBatchNormV3/ReadVariableOp�;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�(model/batch_normalization/ReadVariableOp�*model/batch_normalization/ReadVariableOp_1�;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�*model/batch_normalization_1/ReadVariableOp�,model/batch_normalization_1/ReadVariableOp_1�4model/batch_normalization_2/batchnorm/ReadVariableOp�6model/batch_normalization_2/batchnorm/ReadVariableOp_1�6model/batch_normalization_2/batchnorm/ReadVariableOp_2�8model/batch_normalization_2/batchnorm/mul/ReadVariableOp�#model/conv2d/BiasAdd/ReadVariableOp�"model/conv2d/Conv2D/ReadVariableOp�%model/conv2d_1/BiasAdd/ReadVariableOp�$model/conv2d_1/Conv2D/ReadVariableOp�*model/custom_activation/sub/ReadVariableOp�,model/custom_activation/sub_1/ReadVariableOp�-model/custom_activation/sub_10/ReadVariableOp�-model/custom_activation/sub_11/ReadVariableOp�-model/custom_activation/sub_12/ReadVariableOp�-model/custom_activation/sub_13/ReadVariableOp�-model/custom_activation/sub_14/ReadVariableOp�-model/custom_activation/sub_15/ReadVariableOp�-model/custom_activation/sub_16/ReadVariableOp�-model/custom_activation/sub_17/ReadVariableOp�-model/custom_activation/sub_18/ReadVariableOp�-model/custom_activation/sub_19/ReadVariableOp�,model/custom_activation/sub_2/ReadVariableOp�,model/custom_activation/sub_3/ReadVariableOp�,model/custom_activation/sub_4/ReadVariableOp�,model/custom_activation/sub_5/ReadVariableOp�,model/custom_activation/sub_6/ReadVariableOp�,model/custom_activation/sub_7/ReadVariableOp�,model/custom_activation/sub_8/ReadVariableOp�,model/custom_activation/sub_9/ReadVariableOp�.model/custom_activation/truediv/ReadVariableOp�0model/custom_activation/truediv_1/ReadVariableOp�1model/custom_activation/truediv_10/ReadVariableOp�1model/custom_activation/truediv_11/ReadVariableOp�1model/custom_activation/truediv_12/ReadVariableOp�1model/custom_activation/truediv_13/ReadVariableOp�1model/custom_activation/truediv_14/ReadVariableOp�1model/custom_activation/truediv_15/ReadVariableOp�1model/custom_activation/truediv_16/ReadVariableOp�1model/custom_activation/truediv_17/ReadVariableOp�1model/custom_activation/truediv_18/ReadVariableOp�1model/custom_activation/truediv_19/ReadVariableOp�0model/custom_activation/truediv_2/ReadVariableOp�0model/custom_activation/truediv_3/ReadVariableOp�0model/custom_activation/truediv_4/ReadVariableOp�0model/custom_activation/truediv_5/ReadVariableOp�0model/custom_activation/truediv_6/ReadVariableOp�0model/custom_activation/truediv_7/ReadVariableOp�0model/custom_activation/truediv_8/ReadVariableOp�0model/custom_activation/truediv_9/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�model/p_re_lu/ReadVariableOp�
+model/custom_activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2-
+model/custom_activation/strided_slice/stack�
-model/custom_activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2/
-model/custom_activation/strided_slice/stack_1�
-model/custom_activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2/
-model/custom_activation/strided_slice/stack_2�
%model/custom_activation/strided_sliceStridedSliceinput_14model/custom_activation/strided_slice/stack:output:06model/custom_activation/strided_slice/stack_1:output:06model/custom_activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2'
%model/custom_activation/strided_slice�
*model/custom_activation/sub/ReadVariableOpReadVariableOp3model_custom_activation_sub_readvariableop_resource*
_output_shapes
: *
dtype02,
*model/custom_activation/sub/ReadVariableOp�
model/custom_activation/subSub.model/custom_activation/strided_slice:output:02model/custom_activation/sub/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub�
.model/custom_activation/truediv/ReadVariableOpReadVariableOp7model_custom_activation_truediv_readvariableop_resource*
_output_shapes
: *
dtype020
.model/custom_activation/truediv/ReadVariableOp�
model/custom_activation/truedivRealDivmodel/custom_activation/sub:z:06model/custom_activation/truediv/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/truediv�
model/custom_activation/TanhTanh#model/custom_activation/truediv:z:0*
T0*'
_output_shapes
:���������2
model/custom_activation/Tanh�
-model/custom_activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2/
-model/custom_activation/strided_slice_1/stack�
/model/custom_activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_1/stack_1�
/model/custom_activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_1/stack_2�
'model/custom_activation/strided_slice_1StridedSliceinput_16model/custom_activation/strided_slice_1/stack:output:08model/custom_activation/strided_slice_1/stack_1:output:08model/custom_activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_1�
,model/custom_activation/sub_1/ReadVariableOpReadVariableOp5model_custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_1/ReadVariableOp�
model/custom_activation/sub_1Sub0model/custom_activation/strided_slice_1:output:04model/custom_activation/sub_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_1�
0model/custom_activation/truediv_1/ReadVariableOpReadVariableOp9model_custom_activation_truediv_1_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_1/ReadVariableOp�
!model/custom_activation/truediv_1RealDiv!model/custom_activation/sub_1:z:08model/custom_activation/truediv_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_1�
model/custom_activation/Tanh_1Tanh%model/custom_activation/truediv_1:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_1�
-model/custom_activation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2/
-model/custom_activation/strided_slice_2/stack�
/model/custom_activation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_2/stack_1�
/model/custom_activation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_2/stack_2�
'model/custom_activation/strided_slice_2StridedSliceinput_16model/custom_activation/strided_slice_2/stack:output:08model/custom_activation/strided_slice_2/stack_1:output:08model/custom_activation/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_2�
,model/custom_activation/sub_2/ReadVariableOpReadVariableOp5model_custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_2/ReadVariableOp�
model/custom_activation/sub_2Sub0model/custom_activation/strided_slice_2:output:04model/custom_activation/sub_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_2�
0model/custom_activation/truediv_2/ReadVariableOpReadVariableOp9model_custom_activation_truediv_1_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_2/ReadVariableOp�
!model/custom_activation/truediv_2RealDiv!model/custom_activation/sub_2:z:08model/custom_activation/truediv_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_2�
model/custom_activation/Tanh_2Tanh%model/custom_activation/truediv_2:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_2�
-model/custom_activation/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2/
-model/custom_activation/strided_slice_3/stack�
/model/custom_activation/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_3/stack_1�
/model/custom_activation/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_3/stack_2�
'model/custom_activation/strided_slice_3StridedSliceinput_16model/custom_activation/strided_slice_3/stack:output:08model/custom_activation/strided_slice_3/stack_1:output:08model/custom_activation/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_3�
,model/custom_activation/sub_3/ReadVariableOpReadVariableOp5model_custom_activation_sub_3_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_3/ReadVariableOp�
model/custom_activation/sub_3Sub0model/custom_activation/strided_slice_3:output:04model/custom_activation/sub_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_3�
0model/custom_activation/truediv_3/ReadVariableOpReadVariableOp9model_custom_activation_truediv_3_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_3/ReadVariableOp�
!model/custom_activation/truediv_3RealDiv!model/custom_activation/sub_3:z:08model/custom_activation/truediv_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_3�
model/custom_activation/Tanh_3Tanh%model/custom_activation/truediv_3:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_3�
-model/custom_activation/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2/
-model/custom_activation/strided_slice_4/stack�
/model/custom_activation/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_4/stack_1�
/model/custom_activation/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_4/stack_2�
'model/custom_activation/strided_slice_4StridedSliceinput_16model/custom_activation/strided_slice_4/stack:output:08model/custom_activation/strided_slice_4/stack_1:output:08model/custom_activation/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_4�
,model/custom_activation/sub_4/ReadVariableOpReadVariableOp5model_custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_4/ReadVariableOp�
model/custom_activation/sub_4Sub0model/custom_activation/strided_slice_4:output:04model/custom_activation/sub_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_4�
0model/custom_activation/truediv_4/ReadVariableOpReadVariableOp9model_custom_activation_truediv_4_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_4/ReadVariableOp�
!model/custom_activation/truediv_4RealDiv!model/custom_activation/sub_4:z:08model/custom_activation/truediv_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_4�
model/custom_activation/Tanh_4Tanh%model/custom_activation/truediv_4:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_4�
-model/custom_activation/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2/
-model/custom_activation/strided_slice_5/stack�
/model/custom_activation/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_5/stack_1�
/model/custom_activation/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_5/stack_2�
'model/custom_activation/strided_slice_5StridedSliceinput_16model/custom_activation/strided_slice_5/stack:output:08model/custom_activation/strided_slice_5/stack_1:output:08model/custom_activation/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_5�
,model/custom_activation/sub_5/ReadVariableOpReadVariableOp5model_custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_5/ReadVariableOp�
model/custom_activation/sub_5Sub0model/custom_activation/strided_slice_5:output:04model/custom_activation/sub_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_5�
0model/custom_activation/truediv_5/ReadVariableOpReadVariableOp9model_custom_activation_truediv_4_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_5/ReadVariableOp�
!model/custom_activation/truediv_5RealDiv!model/custom_activation/sub_5:z:08model/custom_activation/truediv_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_5�
model/custom_activation/Tanh_5Tanh%model/custom_activation/truediv_5:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_5�
-model/custom_activation/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2/
-model/custom_activation/strided_slice_6/stack�
/model/custom_activation/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_6/stack_1�
/model/custom_activation/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_6/stack_2�
'model/custom_activation/strided_slice_6StridedSliceinput_16model/custom_activation/strided_slice_6/stack:output:08model/custom_activation/strided_slice_6/stack_1:output:08model/custom_activation/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_6�
,model/custom_activation/sub_6/ReadVariableOpReadVariableOp5model_custom_activation_sub_6_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_6/ReadVariableOp�
model/custom_activation/sub_6Sub0model/custom_activation/strided_slice_6:output:04model/custom_activation/sub_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_6�
0model/custom_activation/truediv_6/ReadVariableOpReadVariableOp9model_custom_activation_truediv_6_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_6/ReadVariableOp�
!model/custom_activation/truediv_6RealDiv!model/custom_activation/sub_6:z:08model/custom_activation/truediv_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_6�
model/custom_activation/Tanh_6Tanh%model/custom_activation/truediv_6:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_6�
-model/custom_activation/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2/
-model/custom_activation/strided_slice_7/stack�
/model/custom_activation/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_7/stack_1�
/model/custom_activation/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_7/stack_2�
'model/custom_activation/strided_slice_7StridedSliceinput_16model/custom_activation/strided_slice_7/stack:output:08model/custom_activation/strided_slice_7/stack_1:output:08model/custom_activation/strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_7�
,model/custom_activation/sub_7/ReadVariableOpReadVariableOp5model_custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_7/ReadVariableOp�
model/custom_activation/sub_7Sub0model/custom_activation/strided_slice_7:output:04model/custom_activation/sub_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_7�
0model/custom_activation/truediv_7/ReadVariableOpReadVariableOp9model_custom_activation_truediv_7_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_7/ReadVariableOp�
!model/custom_activation/truediv_7RealDiv!model/custom_activation/sub_7:z:08model/custom_activation/truediv_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_7�
model/custom_activation/Tanh_7Tanh%model/custom_activation/truediv_7:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_7�
-model/custom_activation/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2/
-model/custom_activation/strided_slice_8/stack�
/model/custom_activation/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_8/stack_1�
/model/custom_activation/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_8/stack_2�
'model/custom_activation/strided_slice_8StridedSliceinput_16model/custom_activation/strided_slice_8/stack:output:08model/custom_activation/strided_slice_8/stack_1:output:08model/custom_activation/strided_slice_8/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_8�
,model/custom_activation/sub_8/ReadVariableOpReadVariableOp5model_custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_8/ReadVariableOp�
model/custom_activation/sub_8Sub0model/custom_activation/strided_slice_8:output:04model/custom_activation/sub_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_8�
0model/custom_activation/truediv_8/ReadVariableOpReadVariableOp9model_custom_activation_truediv_7_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_8/ReadVariableOp�
!model/custom_activation/truediv_8RealDiv!model/custom_activation/sub_8:z:08model/custom_activation/truediv_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_8�
model/custom_activation/Tanh_8Tanh%model/custom_activation/truediv_8:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_8�
-model/custom_activation/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2/
-model/custom_activation/strided_slice_9/stack�
/model/custom_activation/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           21
/model/custom_activation/strided_slice_9/stack_1�
/model/custom_activation/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         21
/model/custom_activation/strided_slice_9/stack_2�
'model/custom_activation/strided_slice_9StridedSliceinput_16model/custom_activation/strided_slice_9/stack:output:08model/custom_activation/strided_slice_9/stack_1:output:08model/custom_activation/strided_slice_9/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2)
'model/custom_activation/strided_slice_9�
,model/custom_activation/sub_9/ReadVariableOpReadVariableOp5model_custom_activation_sub_9_readvariableop_resource*
_output_shapes
: *
dtype02.
,model/custom_activation/sub_9/ReadVariableOp�
model/custom_activation/sub_9Sub0model/custom_activation/strided_slice_9:output:04model/custom_activation/sub_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/custom_activation/sub_9�
0model/custom_activation/truediv_9/ReadVariableOpReadVariableOp9model_custom_activation_truediv_9_readvariableop_resource*
_output_shapes
: *
dtype022
0model/custom_activation/truediv_9/ReadVariableOp�
!model/custom_activation/truediv_9RealDiv!model/custom_activation/sub_9:z:08model/custom_activation/truediv_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!model/custom_activation/truediv_9�
model/custom_activation/Tanh_9Tanh%model/custom_activation/truediv_9:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Tanh_9�
model/custom_activation/stackPack model/custom_activation/Tanh:y:0"model/custom_activation/Tanh_1:y:0"model/custom_activation/Tanh_2:y:0"model/custom_activation/Tanh_3:y:0"model/custom_activation/Tanh_4:y:0"model/custom_activation/Tanh_5:y:0"model/custom_activation/Tanh_6:y:0"model/custom_activation/Tanh_7:y:0"model/custom_activation/Tanh_8:y:0"model/custom_activation/Tanh_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2
model/custom_activation/stack�
.model/custom_activation/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*!
valueB"            20
.model/custom_activation/strided_slice_10/stack�
0model/custom_activation/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_10/stack_1�
0model/custom_activation/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_10/stack_2�
(model/custom_activation/strided_slice_10StridedSlice&model/custom_activation/stack:output:07model/custom_activation/strided_slice_10/stack:output:09model/custom_activation/strided_slice_10/stack_1:output:09model/custom_activation/strided_slice_10/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_10�
%model/custom_activation/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2'
%model/custom_activation/Reshape/shape�
model/custom_activation/ReshapeReshape1model/custom_activation/strided_slice_10:output:0.model/custom_activation/Reshape/shape:output:0*
T0*+
_output_shapes
:���������
2!
model/custom_activation/Reshape�
.model/custom_activation/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.model/custom_activation/strided_slice_11/stack�
0model/custom_activation/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_11/stack_1�
0model/custom_activation/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_11/stack_2�
(model/custom_activation/strided_slice_11StridedSlice&model/custom_activation/stack:output:07model/custom_activation/strided_slice_11/stack:output:09model/custom_activation/strided_slice_11/stack_1:output:09model/custom_activation/strided_slice_11/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_11�
'model/custom_activation/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2)
'model/custom_activation/Reshape_1/shape�
!model/custom_activation/Reshape_1Reshape1model/custom_activation/strided_slice_11:output:00model/custom_activation/Reshape_1/shape:output:0*
T0*+
_output_shapes
:���������
2#
!model/custom_activation/Reshape_1�
#model/custom_activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#model/custom_activation/concat/axis�
model/custom_activation/concatConcatV2*model/custom_activation/Reshape_1:output:0&model/custom_activation/stack:output:0(model/custom_activation/Reshape:output:0,model/custom_activation/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������
2 
model/custom_activation/concat�
'model/custom_activation/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2)
'model/custom_activation/Reshape_2/shape�
!model/custom_activation/Reshape_2Reshape'model/custom_activation/concat:output:00model/custom_activation/Reshape_2/shape:output:0*
T0*/
_output_shapes
:���������
2#
!model/custom_activation/Reshape_2�
.model/custom_activation/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*!
valueB"            20
.model/custom_activation/strided_slice_12/stack�
0model/custom_activation/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_12/stack_1�
0model/custom_activation/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_12/stack_2�
(model/custom_activation/strided_slice_12StridedSliceinput_17model/custom_activation/strided_slice_12/stack:output:09model/custom_activation/strided_slice_12/stack_1:output:09model/custom_activation/strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_12�
-model/custom_activation/sub_10/ReadVariableOpReadVariableOp3model_custom_activation_sub_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_10/ReadVariableOp�
model/custom_activation/sub_10Sub1model/custom_activation/strided_slice_12:output:05model/custom_activation/sub_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_10�
1model/custom_activation/truediv_10/ReadVariableOpReadVariableOp:model_custom_activation_truediv_10_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_10/ReadVariableOp�
"model/custom_activation/truediv_10RealDiv"model/custom_activation/sub_10:z:09model/custom_activation/truediv_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_10�
model/custom_activation/ReluRelu&model/custom_activation/truediv_10:z:0*
T0*'
_output_shapes
:���������2
model/custom_activation/Relu�
model/custom_activation/Log1pLog1p*model/custom_activation/Relu:activations:0*
T0*'
_output_shapes
:���������2
model/custom_activation/Log1p�
.model/custom_activation/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.model/custom_activation/strided_slice_13/stack�
0model/custom_activation/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_13/stack_1�
0model/custom_activation/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_13/stack_2�
(model/custom_activation/strided_slice_13StridedSliceinput_17model/custom_activation/strided_slice_13/stack:output:09model/custom_activation/strided_slice_13/stack_1:output:09model/custom_activation/strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_13�
-model/custom_activation/sub_11/ReadVariableOpReadVariableOp5model_custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_11/ReadVariableOp�
model/custom_activation/sub_11Sub1model/custom_activation/strided_slice_13:output:05model/custom_activation/sub_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_11�
1model/custom_activation/truediv_11/ReadVariableOpReadVariableOp:model_custom_activation_truediv_11_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_11/ReadVariableOp�
"model/custom_activation/truediv_11RealDiv"model/custom_activation/sub_11:z:09model/custom_activation/truediv_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_11�
model/custom_activation/Relu_1Relu&model/custom_activation/truediv_11:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_1�
model/custom_activation/Log1p_1Log1p,model/custom_activation/Relu_1:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_1�
.model/custom_activation/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.model/custom_activation/strided_slice_14/stack�
0model/custom_activation/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_14/stack_1�
0model/custom_activation/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_14/stack_2�
(model/custom_activation/strided_slice_14StridedSliceinput_17model/custom_activation/strided_slice_14/stack:output:09model/custom_activation/strided_slice_14/stack_1:output:09model/custom_activation/strided_slice_14/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_14�
-model/custom_activation/sub_12/ReadVariableOpReadVariableOp5model_custom_activation_sub_1_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_12/ReadVariableOp�
model/custom_activation/sub_12Sub1model/custom_activation/strided_slice_14:output:05model/custom_activation/sub_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_12�
1model/custom_activation/truediv_12/ReadVariableOpReadVariableOp:model_custom_activation_truediv_11_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_12/ReadVariableOp�
"model/custom_activation/truediv_12RealDiv"model/custom_activation/sub_12:z:09model/custom_activation/truediv_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_12�
model/custom_activation/Relu_2Relu&model/custom_activation/truediv_12:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_2�
model/custom_activation/Log1p_2Log1p,model/custom_activation/Relu_2:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_2�
.model/custom_activation/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*!
valueB"            20
.model/custom_activation/strided_slice_15/stack�
0model/custom_activation/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_15/stack_1�
0model/custom_activation/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_15/stack_2�
(model/custom_activation/strided_slice_15StridedSliceinput_17model/custom_activation/strided_slice_15/stack:output:09model/custom_activation/strided_slice_15/stack_1:output:09model/custom_activation/strided_slice_15/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_15�
-model/custom_activation/sub_13/ReadVariableOpReadVariableOp5model_custom_activation_sub_3_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_13/ReadVariableOp�
model/custom_activation/sub_13Sub1model/custom_activation/strided_slice_15:output:05model/custom_activation/sub_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_13�
1model/custom_activation/truediv_13/ReadVariableOpReadVariableOp:model_custom_activation_truediv_13_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_13/ReadVariableOp�
"model/custom_activation/truediv_13RealDiv"model/custom_activation/sub_13:z:09model/custom_activation/truediv_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_13�
model/custom_activation/Relu_3Relu&model/custom_activation/truediv_13:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_3�
model/custom_activation/Log1p_3Log1p,model/custom_activation/Relu_3:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_3�
.model/custom_activation/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.model/custom_activation/strided_slice_16/stack�
0model/custom_activation/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_16/stack_1�
0model/custom_activation/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_16/stack_2�
(model/custom_activation/strided_slice_16StridedSliceinput_17model/custom_activation/strided_slice_16/stack:output:09model/custom_activation/strided_slice_16/stack_1:output:09model/custom_activation/strided_slice_16/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_16�
-model/custom_activation/sub_14/ReadVariableOpReadVariableOp5model_custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_14/ReadVariableOp�
model/custom_activation/sub_14Sub1model/custom_activation/strided_slice_16:output:05model/custom_activation/sub_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_14�
1model/custom_activation/truediv_14/ReadVariableOpReadVariableOp:model_custom_activation_truediv_14_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_14/ReadVariableOp�
"model/custom_activation/truediv_14RealDiv"model/custom_activation/sub_14:z:09model/custom_activation/truediv_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_14�
model/custom_activation/Relu_4Relu&model/custom_activation/truediv_14:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_4�
model/custom_activation/Log1p_4Log1p,model/custom_activation/Relu_4:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_4�
.model/custom_activation/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.model/custom_activation/strided_slice_17/stack�
0model/custom_activation/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_17/stack_1�
0model/custom_activation/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_17/stack_2�
(model/custom_activation/strided_slice_17StridedSliceinput_17model/custom_activation/strided_slice_17/stack:output:09model/custom_activation/strided_slice_17/stack_1:output:09model/custom_activation/strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_17�
-model/custom_activation/sub_15/ReadVariableOpReadVariableOp5model_custom_activation_sub_4_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_15/ReadVariableOp�
model/custom_activation/sub_15Sub1model/custom_activation/strided_slice_17:output:05model/custom_activation/sub_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_15�
1model/custom_activation/truediv_15/ReadVariableOpReadVariableOp:model_custom_activation_truediv_14_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_15/ReadVariableOp�
"model/custom_activation/truediv_15RealDiv"model/custom_activation/sub_15:z:09model/custom_activation/truediv_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_15�
model/custom_activation/Relu_5Relu&model/custom_activation/truediv_15:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_5�
model/custom_activation/Log1p_5Log1p,model/custom_activation/Relu_5:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_5�
.model/custom_activation/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*!
valueB"            20
.model/custom_activation/strided_slice_18/stack�
0model/custom_activation/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_18/stack_1�
0model/custom_activation/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_18/stack_2�
(model/custom_activation/strided_slice_18StridedSliceinput_17model/custom_activation/strided_slice_18/stack:output:09model/custom_activation/strided_slice_18/stack_1:output:09model/custom_activation/strided_slice_18/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_18�
-model/custom_activation/sub_16/ReadVariableOpReadVariableOp5model_custom_activation_sub_6_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_16/ReadVariableOp�
model/custom_activation/sub_16Sub1model/custom_activation/strided_slice_18:output:05model/custom_activation/sub_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_16�
1model/custom_activation/truediv_16/ReadVariableOpReadVariableOp:model_custom_activation_truediv_16_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_16/ReadVariableOp�
"model/custom_activation/truediv_16RealDiv"model/custom_activation/sub_16:z:09model/custom_activation/truediv_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_16�
model/custom_activation/Relu_6Relu&model/custom_activation/truediv_16:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_6�
model/custom_activation/Log1p_6Log1p,model/custom_activation/Relu_6:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_6�
.model/custom_activation/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.model/custom_activation/strided_slice_19/stack�
0model/custom_activation/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_19/stack_1�
0model/custom_activation/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_19/stack_2�
(model/custom_activation/strided_slice_19StridedSliceinput_17model/custom_activation/strided_slice_19/stack:output:09model/custom_activation/strided_slice_19/stack_1:output:09model/custom_activation/strided_slice_19/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_19�
-model/custom_activation/sub_17/ReadVariableOpReadVariableOp5model_custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_17/ReadVariableOp�
model/custom_activation/sub_17Sub1model/custom_activation/strided_slice_19:output:05model/custom_activation/sub_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_17�
1model/custom_activation/truediv_17/ReadVariableOpReadVariableOp:model_custom_activation_truediv_17_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_17/ReadVariableOp�
"model/custom_activation/truediv_17RealDiv"model/custom_activation/sub_17:z:09model/custom_activation/truediv_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_17�
model/custom_activation/Relu_7Relu&model/custom_activation/truediv_17:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_7�
model/custom_activation/Log1p_7Log1p,model/custom_activation/Relu_7:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_7�
.model/custom_activation/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.model/custom_activation/strided_slice_20/stack�
0model/custom_activation/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_20/stack_1�
0model/custom_activation/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_20/stack_2�
(model/custom_activation/strided_slice_20StridedSliceinput_17model/custom_activation/strided_slice_20/stack:output:09model/custom_activation/strided_slice_20/stack_1:output:09model/custom_activation/strided_slice_20/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_20�
-model/custom_activation/sub_18/ReadVariableOpReadVariableOp5model_custom_activation_sub_7_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_18/ReadVariableOp�
model/custom_activation/sub_18Sub1model/custom_activation/strided_slice_20:output:05model/custom_activation/sub_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_18�
1model/custom_activation/truediv_18/ReadVariableOpReadVariableOp:model_custom_activation_truediv_17_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_18/ReadVariableOp�
"model/custom_activation/truediv_18RealDiv"model/custom_activation/sub_18:z:09model/custom_activation/truediv_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_18�
model/custom_activation/Relu_8Relu&model/custom_activation/truediv_18:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_8�
model/custom_activation/Log1p_8Log1p,model/custom_activation/Relu_8:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_8�
.model/custom_activation/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*!
valueB"            20
.model/custom_activation/strided_slice_21/stack�
0model/custom_activation/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_21/stack_1�
0model/custom_activation/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_21/stack_2�
(model/custom_activation/strided_slice_21StridedSliceinput_17model/custom_activation/strided_slice_21/stack:output:09model/custom_activation/strided_slice_21/stack_1:output:09model/custom_activation/strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_21�
-model/custom_activation/sub_19/ReadVariableOpReadVariableOp5model_custom_activation_sub_9_readvariableop_resource*
_output_shapes
: *
dtype02/
-model/custom_activation/sub_19/ReadVariableOp�
model/custom_activation/sub_19Sub1model/custom_activation/strided_slice_21:output:05model/custom_activation/sub_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/sub_19�
1model/custom_activation/truediv_19/ReadVariableOpReadVariableOp:model_custom_activation_truediv_19_readvariableop_resource*
_output_shapes
: *
dtype023
1model/custom_activation/truediv_19/ReadVariableOp�
"model/custom_activation/truediv_19RealDiv"model/custom_activation/sub_19:z:09model/custom_activation/truediv_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"model/custom_activation/truediv_19�
model/custom_activation/Relu_9Relu&model/custom_activation/truediv_19:z:0*
T0*'
_output_shapes
:���������2 
model/custom_activation/Relu_9�
model/custom_activation/Log1p_9Log1p,model/custom_activation/Relu_9:activations:0*
T0*'
_output_shapes
:���������2!
model/custom_activation/Log1p_9�
model/custom_activation/stack_1Pack!model/custom_activation/Log1p:y:0#model/custom_activation/Log1p_1:y:0#model/custom_activation/Log1p_2:y:0#model/custom_activation/Log1p_3:y:0#model/custom_activation/Log1p_4:y:0#model/custom_activation/Log1p_5:y:0#model/custom_activation/Log1p_6:y:0#model/custom_activation/Log1p_7:y:0#model/custom_activation/Log1p_8:y:0#model/custom_activation/Log1p_9:y:0*
N
*
T0*+
_output_shapes
:���������
*

axis2!
model/custom_activation/stack_1�
.model/custom_activation/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*!
valueB"            20
.model/custom_activation/strided_slice_22/stack�
0model/custom_activation/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_22/stack_1�
0model/custom_activation/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_22/stack_2�
(model/custom_activation/strided_slice_22StridedSlice(model/custom_activation/stack_1:output:07model/custom_activation/strided_slice_22/stack:output:09model/custom_activation/strided_slice_22/stack_1:output:09model/custom_activation/strided_slice_22/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_22�
'model/custom_activation/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2)
'model/custom_activation/Reshape_3/shape�
!model/custom_activation/Reshape_3Reshape1model/custom_activation/strided_slice_22:output:00model/custom_activation/Reshape_3/shape:output:0*
T0*+
_output_shapes
:���������
2#
!model/custom_activation/Reshape_3�
.model/custom_activation/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.model/custom_activation/strided_slice_23/stack�
0model/custom_activation/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           22
0model/custom_activation/strided_slice_23/stack_1�
0model/custom_activation/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0model/custom_activation/strided_slice_23/stack_2�
(model/custom_activation/strided_slice_23StridedSlice(model/custom_activation/stack_1:output:07model/custom_activation/strided_slice_23/stack:output:09model/custom_activation/strided_slice_23/stack_1:output:09model/custom_activation/strided_slice_23/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������
*

begin_mask*
end_mask*
shrink_axis_mask2*
(model/custom_activation/strided_slice_23�
'model/custom_activation/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����
      2)
'model/custom_activation/Reshape_4/shape�
!model/custom_activation/Reshape_4Reshape1model/custom_activation/strided_slice_23:output:00model/custom_activation/Reshape_4/shape:output:0*
T0*+
_output_shapes
:���������
2#
!model/custom_activation/Reshape_4�
%model/custom_activation/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%model/custom_activation/concat_1/axis�
 model/custom_activation/concat_1ConcatV2*model/custom_activation/Reshape_4:output:0(model/custom_activation/stack_1:output:0*model/custom_activation/Reshape_3:output:0.model/custom_activation/concat_1/axis:output:0*
N*
T0*+
_output_shapes
:���������
2"
 model/custom_activation/concat_1�
'model/custom_activation/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����
         2)
'model/custom_activation/Reshape_5/shape�
!model/custom_activation/Reshape_5Reshape)model/custom_activation/concat_1:output:00model/custom_activation/Reshape_5/shape:output:0*
T0*/
_output_shapes
:���������
2#
!model/custom_activation/Reshape_5�
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp�
model/conv2d_1/Conv2DConv2D*model/custom_activation/Reshape_5:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
model/conv2d_1/Conv2D�
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp�
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
model/conv2d_1/BiasAdd�
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"model/conv2d/Conv2D/ReadVariableOp�
model/conv2d/Conv2DConv2D*model/custom_activation/Reshape_2:output:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
model/conv2d/Conv2D�
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp�
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
model/conv2d/BiasAdd�
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/batch_normalization_1/ReadVariableOp�
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02.
,model/batch_normalization_1/ReadVariableOp_1�
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02=
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3model/conv2d_1/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2.
,model/batch_normalization_1/FusedBatchNormV3�
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02*
(model/batch_normalization/ReadVariableOp�
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02,
*model/batch_normalization/ReadVariableOp_1�
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02;
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp�
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3model/conv2d/BiasAdd:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2,
*model/batch_normalization/FusedBatchNormV3�
model/tf.math.tanh_1/TanhTanh0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
model/tf.math.tanh_1/Tanh�
model/tf.nn.relu_1/ReluRelu0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
model/tf.nn.relu_1/Relu�
model/tf.math.tanh/TanhTanh.model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
model/tf.math.tanh/Tanh�
model/tf.nn.relu/ReluRelu.model/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������2
model/tf.nn.relu/Relu�
!model/average_pooling2d_3/AvgPoolAvgPoolmodel/tf.math.tanh_1/Tanh:y:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2#
!model/average_pooling2d_3/AvgPool�
model/max_pooling2d_3/MaxPoolMaxPoolmodel/tf.math.tanh_1/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_3/MaxPool�
!model/average_pooling2d_2/AvgPoolAvgPool%model/tf.nn.relu_1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2#
!model/average_pooling2d_2/AvgPool�
model/max_pooling2d_2/MaxPoolMaxPool%model/tf.nn.relu_1/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_2/MaxPool�
!model/average_pooling2d_1/AvgPoolAvgPoolmodel/tf.math.tanh/Tanh:y:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2#
!model/average_pooling2d_1/AvgPool�
model/max_pooling2d_1/MaxPoolMaxPoolmodel/tf.math.tanh/Tanh:y:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool�
model/average_pooling2d/AvgPoolAvgPool#model/tf.nn.relu/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2!
model/average_pooling2d/AvgPool�
model/max_pooling2d/MaxPoolMaxPool#model/tf.nn.relu/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten/Const�
model/flatten/ReshapeReshape$model/max_pooling2d/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten/Reshape
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten_1/Const�
model/flatten_1/ReshapeReshape(model/average_pooling2d/AvgPool:output:0model/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten_1/Reshape
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten_2/Const�
model/flatten_2/ReshapeReshape&model/max_pooling2d_1/MaxPool:output:0model/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten_2/Reshape
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten_3/Const�
model/flatten_3/ReshapeReshape*model/average_pooling2d_1/AvgPool:output:0model/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten_3/Reshape
model/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten_4/Const�
model/flatten_4/ReshapeReshape&model/max_pooling2d_2/MaxPool:output:0model/flatten_4/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten_4/Reshape
model/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten_5/Const�
model/flatten_5/ReshapeReshape*model/average_pooling2d_2/AvgPool:output:0model/flatten_5/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten_5/Reshape
model/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten_6/Const�
model/flatten_6/ReshapeReshape&model/max_pooling2d_3/MaxPool:output:0model/flatten_6/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten_6/Reshape
model/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model/flatten_7/Const�
model/flatten_7/ReshapeReshape*model/average_pooling2d_3/AvgPool:output:0model/flatten_7/Const:output:0*
T0*(
_output_shapes
:����������2
model/flatten_7/Reshape�
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis�
model/concatenate/concatConcatV2model/flatten/Reshape:output:0 model/flatten_1/Reshape:output:0 model/flatten_2/Reshape:output:0 model/flatten_3/Reshape:output:0 model/flatten_4/Reshape:output:0 model/flatten_5/Reshape:output:0 model/flatten_6/Reshape:output:0 model/flatten_7/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
model/concatenate/concat�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense/BiasAdd�
4model/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype026
4model/batch_normalization_2/batchnorm/ReadVariableOp�
+model/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+model/batch_normalization_2/batchnorm/add/y�
)model/batch_normalization_2/batchnorm/addAddV2<model/batch_normalization_2/batchnorm/ReadVariableOp:value:04model/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2+
)model/batch_normalization_2/batchnorm/add�
+model/batch_normalization_2/batchnorm/RsqrtRsqrt-model/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes	
:�2-
+model/batch_normalization_2/batchnorm/Rsqrt�
8model/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8model/batch_normalization_2/batchnorm/mul/ReadVariableOp�
)model/batch_normalization_2/batchnorm/mulMul/model/batch_normalization_2/batchnorm/Rsqrt:y:0@model/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2+
)model/batch_normalization_2/batchnorm/mul�
+model/batch_normalization_2/batchnorm/mul_1Mulmodel/dense/BiasAdd:output:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2-
+model/batch_normalization_2/batchnorm/mul_1�
6model/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6model/batch_normalization_2/batchnorm/ReadVariableOp_1�
+model/batch_normalization_2/batchnorm/mul_2Mul>model/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2-
+model/batch_normalization_2/batchnorm/mul_2�
6model/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype028
6model/batch_normalization_2/batchnorm/ReadVariableOp_2�
)model/batch_normalization_2/batchnorm/subSub>model/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2+
)model/batch_normalization_2/batchnorm/sub�
+model/batch_normalization_2/batchnorm/add_1AddV2/model/batch_normalization_2/batchnorm/mul_1:z:0-model/batch_normalization_2/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2-
+model/batch_normalization_2/batchnorm/add_1�
model/p_re_lu/ReluRelu/model/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
model/p_re_lu/Relu�
model/p_re_lu/ReadVariableOpReadVariableOp%model_p_re_lu_readvariableop_resource*
_output_shapes	
:�*
dtype02
model/p_re_lu/ReadVariableOpy
model/p_re_lu/NegNeg$model/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
model/p_re_lu/Neg�
model/p_re_lu/Neg_1Neg/model/batch_normalization_2/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
model/p_re_lu/Neg_1�
model/p_re_lu/Relu_1Relumodel/p_re_lu/Neg_1:y:0*
T0*(
_output_shapes
:����������2
model/p_re_lu/Relu_1�
model/p_re_lu/mulMulmodel/p_re_lu/Neg:y:0"model/p_re_lu/Relu_1:activations:0*
T0*(
_output_shapes
:����������2
model/p_re_lu/mul�
model/p_re_lu/addAddV2 model/p_re_lu/Relu:activations:0model/p_re_lu/mul:z:0*
T0*(
_output_shapes
:����������2
model/p_re_lu/add�
model/dropout/IdentityIdentitymodel/p_re_lu/add:z:0*
T0*(
_output_shapes
:����������2
model/dropout/Identity�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_1/BiasAdd�
model/dense_1/SoftmaxSoftmaxmodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/dense_1/Softmax�
IdentityIdentitymodel/dense_1/Softmax:softmax:0:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_15^model/batch_normalization_2/batchnorm/ReadVariableOp7^model/batch_normalization_2/batchnorm/ReadVariableOp_17^model/batch_normalization_2/batchnorm/ReadVariableOp_29^model/batch_normalization_2/batchnorm/mul/ReadVariableOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp+^model/custom_activation/sub/ReadVariableOp-^model/custom_activation/sub_1/ReadVariableOp.^model/custom_activation/sub_10/ReadVariableOp.^model/custom_activation/sub_11/ReadVariableOp.^model/custom_activation/sub_12/ReadVariableOp.^model/custom_activation/sub_13/ReadVariableOp.^model/custom_activation/sub_14/ReadVariableOp.^model/custom_activation/sub_15/ReadVariableOp.^model/custom_activation/sub_16/ReadVariableOp.^model/custom_activation/sub_17/ReadVariableOp.^model/custom_activation/sub_18/ReadVariableOp.^model/custom_activation/sub_19/ReadVariableOp-^model/custom_activation/sub_2/ReadVariableOp-^model/custom_activation/sub_3/ReadVariableOp-^model/custom_activation/sub_4/ReadVariableOp-^model/custom_activation/sub_5/ReadVariableOp-^model/custom_activation/sub_6/ReadVariableOp-^model/custom_activation/sub_7/ReadVariableOp-^model/custom_activation/sub_8/ReadVariableOp-^model/custom_activation/sub_9/ReadVariableOp/^model/custom_activation/truediv/ReadVariableOp1^model/custom_activation/truediv_1/ReadVariableOp2^model/custom_activation/truediv_10/ReadVariableOp2^model/custom_activation/truediv_11/ReadVariableOp2^model/custom_activation/truediv_12/ReadVariableOp2^model/custom_activation/truediv_13/ReadVariableOp2^model/custom_activation/truediv_14/ReadVariableOp2^model/custom_activation/truediv_15/ReadVariableOp2^model/custom_activation/truediv_16/ReadVariableOp2^model/custom_activation/truediv_17/ReadVariableOp2^model/custom_activation/truediv_18/ReadVariableOp2^model/custom_activation/truediv_19/ReadVariableOp1^model/custom_activation/truediv_2/ReadVariableOp1^model/custom_activation/truediv_3/ReadVariableOp1^model/custom_activation/truediv_4/ReadVariableOp1^model/custom_activation/truediv_5/ReadVariableOp1^model/custom_activation/truediv_6/ReadVariableOp1^model/custom_activation/truediv_7/ReadVariableOp1^model/custom_activation/truediv_8/ReadVariableOp1^model/custom_activation/truediv_9/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp^model/p_re_lu/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12l
4model/batch_normalization_2/batchnorm/ReadVariableOp4model/batch_normalization_2/batchnorm/ReadVariableOp2p
6model/batch_normalization_2/batchnorm/ReadVariableOp_16model/batch_normalization_2/batchnorm/ReadVariableOp_12p
6model/batch_normalization_2/batchnorm/ReadVariableOp_26model/batch_normalization_2/batchnorm/ReadVariableOp_22t
8model/batch_normalization_2/batchnorm/mul/ReadVariableOp8model/batch_normalization_2/batchnorm/mul/ReadVariableOp2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2X
*model/custom_activation/sub/ReadVariableOp*model/custom_activation/sub/ReadVariableOp2\
,model/custom_activation/sub_1/ReadVariableOp,model/custom_activation/sub_1/ReadVariableOp2^
-model/custom_activation/sub_10/ReadVariableOp-model/custom_activation/sub_10/ReadVariableOp2^
-model/custom_activation/sub_11/ReadVariableOp-model/custom_activation/sub_11/ReadVariableOp2^
-model/custom_activation/sub_12/ReadVariableOp-model/custom_activation/sub_12/ReadVariableOp2^
-model/custom_activation/sub_13/ReadVariableOp-model/custom_activation/sub_13/ReadVariableOp2^
-model/custom_activation/sub_14/ReadVariableOp-model/custom_activation/sub_14/ReadVariableOp2^
-model/custom_activation/sub_15/ReadVariableOp-model/custom_activation/sub_15/ReadVariableOp2^
-model/custom_activation/sub_16/ReadVariableOp-model/custom_activation/sub_16/ReadVariableOp2^
-model/custom_activation/sub_17/ReadVariableOp-model/custom_activation/sub_17/ReadVariableOp2^
-model/custom_activation/sub_18/ReadVariableOp-model/custom_activation/sub_18/ReadVariableOp2^
-model/custom_activation/sub_19/ReadVariableOp-model/custom_activation/sub_19/ReadVariableOp2\
,model/custom_activation/sub_2/ReadVariableOp,model/custom_activation/sub_2/ReadVariableOp2\
,model/custom_activation/sub_3/ReadVariableOp,model/custom_activation/sub_3/ReadVariableOp2\
,model/custom_activation/sub_4/ReadVariableOp,model/custom_activation/sub_4/ReadVariableOp2\
,model/custom_activation/sub_5/ReadVariableOp,model/custom_activation/sub_5/ReadVariableOp2\
,model/custom_activation/sub_6/ReadVariableOp,model/custom_activation/sub_6/ReadVariableOp2\
,model/custom_activation/sub_7/ReadVariableOp,model/custom_activation/sub_7/ReadVariableOp2\
,model/custom_activation/sub_8/ReadVariableOp,model/custom_activation/sub_8/ReadVariableOp2\
,model/custom_activation/sub_9/ReadVariableOp,model/custom_activation/sub_9/ReadVariableOp2`
.model/custom_activation/truediv/ReadVariableOp.model/custom_activation/truediv/ReadVariableOp2d
0model/custom_activation/truediv_1/ReadVariableOp0model/custom_activation/truediv_1/ReadVariableOp2f
1model/custom_activation/truediv_10/ReadVariableOp1model/custom_activation/truediv_10/ReadVariableOp2f
1model/custom_activation/truediv_11/ReadVariableOp1model/custom_activation/truediv_11/ReadVariableOp2f
1model/custom_activation/truediv_12/ReadVariableOp1model/custom_activation/truediv_12/ReadVariableOp2f
1model/custom_activation/truediv_13/ReadVariableOp1model/custom_activation/truediv_13/ReadVariableOp2f
1model/custom_activation/truediv_14/ReadVariableOp1model/custom_activation/truediv_14/ReadVariableOp2f
1model/custom_activation/truediv_15/ReadVariableOp1model/custom_activation/truediv_15/ReadVariableOp2f
1model/custom_activation/truediv_16/ReadVariableOp1model/custom_activation/truediv_16/ReadVariableOp2f
1model/custom_activation/truediv_17/ReadVariableOp1model/custom_activation/truediv_17/ReadVariableOp2f
1model/custom_activation/truediv_18/ReadVariableOp1model/custom_activation/truediv_18/ReadVariableOp2f
1model/custom_activation/truediv_19/ReadVariableOp1model/custom_activation/truediv_19/ReadVariableOp2d
0model/custom_activation/truediv_2/ReadVariableOp0model/custom_activation/truediv_2/ReadVariableOp2d
0model/custom_activation/truediv_3/ReadVariableOp0model/custom_activation/truediv_3/ReadVariableOp2d
0model/custom_activation/truediv_4/ReadVariableOp0model/custom_activation/truediv_4/ReadVariableOp2d
0model/custom_activation/truediv_5/ReadVariableOp0model/custom_activation/truediv_5/ReadVariableOp2d
0model/custom_activation/truediv_6/ReadVariableOp0model/custom_activation/truediv_6/ReadVariableOp2d
0model/custom_activation/truediv_7/ReadVariableOp0model/custom_activation/truediv_7/ReadVariableOp2d
0model/custom_activation/truediv_8/ReadVariableOp0model/custom_activation/truediv_8/ReadVariableOp2d
0model/custom_activation/truediv_9/ReadVariableOp0model/custom_activation/truediv_9/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2<
model/p_re_lu/ReadVariableOpmodel/p_re_lu/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
__inference_loss_fn_4_5401;
7dense_kernel_regularizer_square_readvariableop_resource
identity��.dense/kernel/Regularizer/Square/ReadVariableOp�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentity dense/kernel/Regularizer/mul:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
�,
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1247

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_6_layer_call_and_return_conditional_losses_2303

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_p_re_lu_layer_call_and_return_conditional_losses_1598

inputs
readvariableop_resource
identity��ReadVariableOpW
ReluReluinputs*
T0*0
_output_shapes
:������������������2
Reluu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOpO
NegNegReadVariableOp:value:0*
T0*
_output_shapes	
:�2
NegX
Neg_1Neginputs*
T0*0
_output_shapes
:������������������2
Neg_1^
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:������������������2
Relu_1c
mulMulNeg:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������2
mulc
addAddV2Relu:activations:0mul:z:0*
T0*(
_output_shapes
:����������2
addm
IdentityIdentityadd:z:0^ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :������������������:2 
ReadVariableOpReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_4_layer_call_and_return_conditional_losses_5087

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�G
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1529

inputs
assignmovingavg_1492
assignmovingavg_1_1498)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/1492*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1492*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/1492*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*'
_class
loc:@AssignMovingAvg/1492*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1492AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*'
_class
loc:@AssignMovingAvg/1492*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/1498*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1498*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/1498*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@AssignMovingAvg_1/1498*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1498AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*)
_class
loc:@AssignMovingAvg_1/1498*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�&
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5011

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2038

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
?__inference_model_layer_call_and_return_conditional_losses_2523
input_1
custom_activation_1911
custom_activation_1913
custom_activation_1915
custom_activation_1917
custom_activation_1919
custom_activation_1921
custom_activation_1923
custom_activation_1925
custom_activation_1927
custom_activation_1929
custom_activation_1931
custom_activation_1933
custom_activation_1935
custom_activation_1937
custom_activation_1939
custom_activation_1941
custom_activation_1943
custom_activation_1945
custom_activation_1947
custom_activation_1949
custom_activation_1951
conv2d_1_1976
conv2d_1_1978
conv2d_2002
conv2d_2004
batch_normalization_1_2095
batch_normalization_1_2097
batch_normalization_1_2099
batch_normalization_1_2101
batch_normalization_2192
batch_normalization_2194
batch_normalization_2196
batch_normalization_2198

dense_2380

dense_2382
batch_normalization_2_2411
batch_normalization_2_2413
batch_normalization_2_2415
batch_normalization_2_2417
p_re_lu_2420
dense_1_2475
dense_1_2477
identity��+batch_normalization/StatefulPartitionedCall�:batch_normalization/beta/Regularizer/Square/ReadVariableOp�;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�-batch_normalization_1/StatefulPartitionedCall�<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�-batch_normalization_2/StatefulPartitionedCall�<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�)custom_activation/StatefulPartitionedCall�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�p_re_lu/StatefulPartitionedCall�
)custom_activation/StatefulPartitionedCallStatefulPartitionedCallinput_1custom_activation_1911custom_activation_1913custom_activation_1915custom_activation_1917custom_activation_1919custom_activation_1921custom_activation_1923custom_activation_1925custom_activation_1927custom_activation_1929custom_activation_1931custom_activation_1933custom_activation_1935custom_activation_1937custom_activation_1939custom_activation_1941custom_activation_1943custom_activation_1945custom_activation_1947custom_activation_1949custom_activation_1951*!
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:���������
:���������
*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_custom_activation_layer_call_and_return_conditional_losses_18602+
)custom_activation/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall2custom_activation/StatefulPartitionedCall:output:1conv2d_1_1976conv2d_1_1978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_19652"
 conv2d_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCall2custom_activation/StatefulPartitionedCall:output:0conv2d_2002conv2d_2004*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_19912 
conv2d/StatefulPartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_2095batch_normalization_1_2097batch_normalization_1_2099batch_normalization_1_2101*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20382/
-batch_normalization_1/StatefulPartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_2192batch_normalization_2194batch_normalization_2196batch_normalization_2198*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_21352-
+batch_normalization/StatefulPartitionedCall�
tf.math.tanh_1/TanhTanh6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.math.tanh_1/Tanh�
tf.nn.relu_1/ReluRelu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.nn.relu_1/Relu�
tf.math.tanh/TanhTanh4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.math.tanh/Tanh�
tf.nn.relu/ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:���������2
tf.nn.relu/Relu�
#average_pooling2d_3/PartitionedCallPartitionedCalltf.math.tanh_1/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13912%
#average_pooling2d_3/PartitionedCall�
max_pooling2d_3/PartitionedCallPartitionedCalltf.math.tanh_1/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_13792!
max_pooling2d_3/PartitionedCall�
#average_pooling2d_2/PartitionedCallPartitionedCalltf.nn.relu_1/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_13672%
#average_pooling2d_2/PartitionedCall�
max_pooling2d_2/PartitionedCallPartitionedCalltf.nn.relu_1/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_13552!
max_pooling2d_2/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCalltf.math.tanh/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_13432%
#average_pooling2d_1/PartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCalltf.math.tanh/Tanh:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_13312!
max_pooling2d_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalltf.nn.relu/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_average_pooling2d_layer_call_and_return_conditional_losses_13192#
!average_pooling2d/PartitionedCall�
max_pooling2d/PartitionedCallPartitionedCalltf.nn.relu/Relu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13072
max_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_22192
flatten/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_22332
flatten_1/PartitionedCall�
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_2_layer_call_and_return_conditional_losses_22472
flatten_2/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_22612
flatten_3/PartitionedCall�
flatten_4/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_4_layer_call_and_return_conditional_losses_22752
flatten_4/PartitionedCall�
flatten_5/PartitionedCallPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_5_layer_call_and_return_conditional_losses_22892
flatten_5/PartitionedCall�
flatten_6/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_6_layer_call_and_return_conditional_losses_23032
flatten_6/PartitionedCall�
flatten_7/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_7_layer_call_and_return_conditional_losses_23172
flatten_7/PartitionedCall�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_23382
concatenate/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_2380
dense_2382*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_23692
dense/StatefulPartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_2_2411batch_normalization_2_2413batch_normalization_2_2415batch_normalization_2_2417*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_15292/
-batch_normalization_2/StatefulPartitionedCall�
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0p_re_lu_2420*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_p_re_lu_layer_call_and_return_conditional_losses_15982!
p_re_lu/StatefulPartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24352!
dropout/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_2475dense_1_2477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_24642!
dense_1/StatefulPartitionedCall�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2192*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
:batch_normalization/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2194*
_output_shapes
:*
dtype02<
:batch_normalization/beta/Regularizer/Square/ReadVariableOp�
+batch_normalization/beta/Regularizer/SquareSquareBbatch_normalization/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+batch_normalization/beta/Regularizer/Square�
*batch_normalization/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batch_normalization/beta/Regularizer/Const�
(batch_normalization/beta/Regularizer/SumSum/batch_normalization/beta/Regularizer/Square:y:03batch_normalization/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/Sum�
*batch_normalization/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*batch_normalization/beta/Regularizer/mul/x�
(batch_normalization/beta/Regularizer/mulMul3batch_normalization/beta/Regularizer/mul/x:output:01batch_normalization/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(batch_normalization/beta/Regularizer/mul�
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_1_2095*
_output_shapes
:*
dtype02?
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_1/gamma/Regularizer/SquareSquareEbatch_normalization_1/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.batch_normalization_1/gamma/Regularizer/Square�
-batch_normalization_1/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_1/gamma/Regularizer/Const�
+batch_normalization_1/gamma/Regularizer/SumSum2batch_normalization_1/gamma/Regularizer/Square:y:06batch_normalization_1/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/Sum�
-batch_normalization_1/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_1/gamma/Regularizer/mul/x�
+batch_normalization_1/gamma/Regularizer/mulMul6batch_normalization_1/gamma/Regularizer/mul/x:output:04batch_normalization_1/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_1/gamma/Regularizer/mul�
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_1_2097*
_output_shapes
:*
dtype02>
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_1/beta/Regularizer/SquareSquareDbatch_normalization_1/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2/
-batch_normalization_1/beta/Regularizer/Square�
,batch_normalization_1/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_1/beta/Regularizer/Const�
*batch_normalization_1/beta/Regularizer/SumSum1batch_normalization_1/beta/Regularizer/Square:y:05batch_normalization_1/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/Sum�
,batch_normalization_1/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_1/beta/Regularizer/mul/x�
*batch_normalization_1/beta/Regularizer/mulMul5batch_normalization_1/beta/Regularizer/mul/x:output:03batch_normalization_1/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_1/beta/Regularizer/mul�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_2380* 
_output_shapes
:
��*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2_2415*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOpReadVariableOpbatch_normalization_2_2417*
_output_shapes	
:�*
dtype02>
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp�
-batch_normalization_2/beta/Regularizer/SquareSquareDbatch_normalization_2/beta/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2/
-batch_normalization_2/beta/Regularizer/Square�
,batch_normalization_2/beta/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,batch_normalization_2/beta/Regularizer/Const�
*batch_normalization_2/beta/Regularizer/SumSum1batch_normalization_2/beta/Regularizer/Square:y:05batch_normalization_2/beta/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/Sum�
,batch_normalization_2/beta/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_2/beta/Regularizer/mul/x�
*batch_normalization_2/beta/Regularizer/mulMul5batch_normalization_2/beta/Regularizer/mul/x:output:03batch_normalization_2/beta/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_2/beta/Regularizer/mul�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall;^batch_normalization/beta/Regularizer/Square/ReadVariableOp<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp.^batch_normalization_1/StatefulPartitionedCall=^batch_normalization_1/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp.^batch_normalization_2/StatefulPartitionedCall=^batch_normalization_2/beta/Regularizer/Square/ReadVariableOp>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*^custom_activation/StatefulPartitionedCall^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2x
:batch_normalization/beta/Regularizer/Square/ReadVariableOp:batch_normalization/beta/Regularizer/Square/ReadVariableOp2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2|
<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp<batch_normalization_1/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_1/gamma/Regularizer/Square/ReadVariableOp2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2|
<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp<batch_normalization_2/beta/Regularizer/Square/ReadVariableOp2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2V
)custom_activation/StatefulPartitionedCall)custom_activation/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
4__inference_batch_normalization_2_layer_call_fn_5299

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_15742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_5357H
Dbatch_normalization_gamma_regularizer_square_readvariableop_resource
identity��;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
;batch_normalization/gamma/Regularizer/Square/ReadVariableOpReadVariableOpDbatch_normalization_gamma_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype02=
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp�
,batch_normalization/gamma/Regularizer/SquareSquareCbatch_normalization/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2.
,batch_normalization/gamma/Regularizer/Square�
+batch_normalization/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+batch_normalization/gamma/Regularizer/Const�
)batch_normalization/gamma/Regularizer/SumSum0batch_normalization/gamma/Regularizer/Square:y:04batch_normalization/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/Sum�
+batch_normalization/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization/gamma/Regularizer/mul/x�
)batch_normalization/gamma/Regularizer/mulMul4batch_normalization/gamma/Regularizer/mul/x:output:02batch_normalization/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)batch_normalization/gamma/Regularizer/mul�
IdentityIdentity-batch_normalization/gamma/Regularizer/mul:z:0<^batch_normalization/gamma/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;batch_normalization/gamma/Regularizer/Square/ReadVariableOp;batch_normalization/gamma/Regularizer/Square/ReadVariableOp
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1307

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_1_layer_call_fn_5024

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1355

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_4324

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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_31012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
_
C__inference_flatten_4_layer_call_and_return_conditional_losses_2275

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
N
2__inference_average_pooling2d_3_layer_call_fn_1397

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_13912
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_1_layer_call_fn_4949

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
D
(__inference_flatten_3_layer_call_fn_5081

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
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_22612
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
i
M__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_1367

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
*__inference_concatenate_layer_call_fn_5150
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_23382
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/4:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/7
�	
�
A__inference_dense_1_layer_call_and_return_conditional_losses_2464

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_3321
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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__wrapped_model_9972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
i
M__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_1343

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
_
C__inference_flatten_5_layer_call_and_return_conditional_losses_5098

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
identityIdentity:output:0*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_5_5412J
Fbatch_normalization_2_gamma_regularizer_square_readvariableop_resource
identity��=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOpReadVariableOpFbatch_normalization_2_gamma_regularizer_square_readvariableop_resource*
_output_shapes	
:�*
dtype02?
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp�
.batch_normalization_2/gamma/Regularizer/SquareSquareEbatch_normalization_2/gamma/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�20
.batch_normalization_2/gamma/Regularizer/Square�
-batch_normalization_2/gamma/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-batch_normalization_2/gamma/Regularizer/Const�
+batch_normalization_2/gamma/Regularizer/SumSum2batch_normalization_2/gamma/Regularizer/Square:y:06batch_normalization_2/gamma/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/Sum�
-batch_normalization_2/gamma/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_2/gamma/Regularizer/mul/x�
+batch_normalization_2/gamma/Regularizer/mulMul6batch_normalization_2/gamma/Regularizer/mul/x:output:04batch_normalization_2/gamma/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+batch_normalization_2/gamma/Regularizer/mul�
IdentityIdentity/batch_normalization_2/gamma/Regularizer/mul:z:0>^batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp=batch_normalization_2/gamma/Regularizer/Square/ReadVariableOp
�
�
4__inference_batch_normalization_1_layer_call_fn_5037

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_20682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
��
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer_with_weights-5
layer-27
layer_with_weights-6
layer-28
layer_with_weights-7
layer-29
layer-30
 layer_with_weights-8
 layer-31
!	optimizer
"trainable_variables
#regularization_losses
$	variables
%	keras_api
&
signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"��
_tf_keras_network��{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "CustomActivation", "config": {"name": "custom_activation", "trainable": true, "dtype": "float32"}, "name": "custom_activation", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["custom_activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["custom_activation", 0, 1, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu", "inbound_nodes": [["batch_normalization", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.tanh", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "name": "tf.math.tanh", "inbound_nodes": [["batch_normalization", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_1", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_1", "inbound_nodes": [["batch_normalization_1", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.tanh_1", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "name": "tf.math.tanh_1", "inbound_nodes": [["batch_normalization_1", 0, 0, {"name": null}]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["tf.nn.relu", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["tf.nn.relu", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["tf.math.tanh", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "average_pooling2d_1", "inbound_nodes": [[["tf.math.tanh", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["tf.nn.relu_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "average_pooling2d_2", "inbound_nodes": [[["tf.nn.relu_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["tf.math.tanh_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "average_pooling2d_3", "inbound_nodes": [[["tf.math.tanh_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["average_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["average_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [{"class_name": "__tuple__", "items": [["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}], ["flatten_4", 0, 0, {}], ["flatten_5", 0, 0, {}], ["flatten_6", 0, 0, {}], ["flatten_7", 0, 0, {}]]}]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.019999999552965164}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["p_re_lu", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 14, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3, 12]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "CustomActivation", "config": {"name": "custom_activation", "trainable": true, "dtype": "float32"}, "name": "custom_activation", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["custom_activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["custom_activation", 0, 1, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu", "inbound_nodes": [["batch_normalization", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.tanh", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "name": "tf.math.tanh", "inbound_nodes": [["batch_normalization", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_1", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_1", "inbound_nodes": [["batch_normalization_1", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.tanh_1", "trainable": true, "dtype": "float32", "function": "math.tanh"}, "name": "tf.math.tanh_1", "inbound_nodes": [["batch_normalization_1", 0, 0, {"name": null}]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["tf.nn.relu", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["tf.nn.relu", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["tf.math.tanh", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "average_pooling2d_1", "inbound_nodes": [[["tf.math.tanh", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["tf.nn.relu_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "average_pooling2d_2", "inbound_nodes": [[["tf.nn.relu_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["tf.math.tanh_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "name": "average_pooling2d_3", "inbound_nodes": [[["tf.math.tanh_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["average_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["average_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [{"class_name": "__tuple__", "items": [["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}], ["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}], ["flatten_4", 0, 0, {}], ["flatten_5", 0, 0, {}], ["flatten_6", 0, 0, {}], ["flatten_7", 0, 0, {}]]}]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.019999999552965164}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["p_re_lu", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 14, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": "CustomLoss", "metrics": [{"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": {"class_name": "ExponentialDecay", "config": {"initial_learning_rate": 0.001, "decay_steps": 5200, "decay_rate": 0.1, "staircase": false, "name": null}}, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�

'mu_t_1

(mu_t_2

)mu_t_3

*mu_t_4
+	sigma_t_1
,	sigma_t_2
-	sigma_t_3
.	sigma_t_4
/	sigma_t_5
0	sigma_t_6
1	sigma_t_7
2	sigma_t_8

3mu_p_1

4mu_p_2

5mu_p_3
6	sigma_p_1
7	sigma_p_2
8	sigma_p_3
9	sigma_p_4
:	sigma_p_5
;	sigma_p_6
<trainable_variables
=regularization_losses
>	variables
?	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "CustomActivation", "name": "custom_activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "custom_activation", "trainable": true, "dtype": "float32"}}
�	

@kernel
Abias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 14, 1]}}
�	

Fkernel
Gbias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 14, 1]}}
�

Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 12, 24]}}
�

Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 12, 24]}}
�
^	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.nn.relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
�
_	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.tanh", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.tanh", "trainable": true, "dtype": "float32", "function": "math.tanh"}}
�
`	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.nn.relu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_1", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
�
a	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.tanh_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.tanh_1", "trainable": true, "dtype": "float32", "function": "math.tanh"}}
�
btrainable_variables
cregularization_losses
d	variables
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
ztrainable_variables
{regularization_losses
|	variables
}	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
~trainable_variables
regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 12]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 12]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": {"class_name": "__tuple__", "items": [{"class_name": "TensorShape", "items": [null, 192]}, {"class_name": "TensorShape", "items": [null, 192]}, {"class_name": "TensorShape", "items": [null, 192]}, {"class_name": "TensorShape", "items": [null, 192]}, {"class_name": "TensorShape", "items": [null, 192]}, {"class_name": "TensorShape", "items": [null, 192]}, {"class_name": "TensorShape", "items": [null, 192]}, {"class_name": "TensorShape", "items": [null, 192]}]}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 768, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.019999999552965164}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1536}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1536]}}
�

	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "gamma_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 768}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
�

�alpha
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "p_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Constant", "config": {"value": 0.25}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 14, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 768}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 768]}}
"
	optimizer
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
@21
A22
F23
G24
M25
N26
V27
W28
�29
�30
�31
�32
�33
�34
�35"
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
@21
A22
F23
G24
M25
N26
O27
P28
V29
W30
X31
Y32
�33
�34
�35
�36
�37
�38
�39
�40
�41"
trackable_list_wrapper
�
"trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
#regularization_losses
�metrics
�layer_metrics
$	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
: 2Variable
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20"
trackable_list_wrapper
 "
trackable_list_wrapper
�
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20"
trackable_list_wrapper
�
<trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
=regularization_losses
�metrics
�layer_metrics
>	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
�
Btrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
Cregularization_losses
�metrics
�layer_metrics
D	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
�
Htrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
Iregularization_losses
�metrics
�layer_metrics
J	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
.
M0
N1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
<
M0
N1
O2
P3"
trackable_list_wrapper
�
Qtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
Rregularization_losses
�metrics
�layer_metrics
S	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
.
V0
W1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
<
V0
W1
X2
Y3"
trackable_list_wrapper
�
Ztrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
[regularization_losses
�metrics
�layer_metrics
\	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
btrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
cregularization_losses
�metrics
�layer_metrics
d	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ftrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
gregularization_losses
�metrics
�layer_metrics
h	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
jtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
kregularization_losses
�metrics
�layer_metrics
l	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ntrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
oregularization_losses
�metrics
�layer_metrics
p	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
sregularization_losses
�metrics
�layer_metrics
t	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
vtrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
wregularization_losses
�metrics
�layer_metrics
x	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ztrainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
{regularization_losses
�metrics
�layer_metrics
|	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
~trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :
��2dense/kernel
:�2
dense/bias
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(�2batch_normalization_2/gamma
):'�2batch_normalization_2/beta
2:0� (2!batch_normalization_2/moving_mean
6:4� (2%batch_normalization_2/moving_variance
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:�2p_re_lu/alpha
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_1/kernel
:2dense_1/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�trainable_variables
�non_trainable_variables
 �layer_regularization_losses
�layers
�regularization_losses
�metrics
�layer_metrics
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
L
O0
P1
X2
Y3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
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
22
23
24
25
26
27
28
29
30
 31"
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
.
O0
P1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
0
�0
�1"
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
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
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
�2�
?__inference_model_layer_call_and_return_conditional_losses_3747
?__inference_model_layer_call_and_return_conditional_losses_4146
?__inference_model_layer_call_and_return_conditional_losses_2685
?__inference_model_layer_call_and_return_conditional_losses_2523�
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
__inference__wrapped_model_997�
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
annotations� **�'
%�"
input_1���������
�2�
$__inference_model_layer_call_fn_2937
$__inference_model_layer_call_fn_3188
$__inference_model_layer_call_fn_4235
$__inference_model_layer_call_fn_4324�
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
K__inference_custom_activation_layer_call_and_return_conditional_losses_4574�
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
0__inference_custom_activation_layer_call_fn_4623�
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
@__inference_conv2d_layer_call_and_return_conditional_losses_4633�
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
%__inference_conv2d_layer_call_fn_4642�
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
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4652�
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
'__inference_conv2d_1_layer_call_fn_4661�
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
�2�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4735
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4705
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4793
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4823�
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
�2�
2__inference_batch_normalization_layer_call_fn_4849
2__inference_batch_normalization_layer_call_fn_4836
2__inference_batch_normalization_layer_call_fn_4761
2__inference_batch_normalization_layer_call_fn_4748�
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
�2�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4923
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4893
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5011
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4981�
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
�2�
4__inference_batch_normalization_1_layer_call_fn_4949
4__inference_batch_normalization_1_layer_call_fn_5024
4__inference_batch_normalization_1_layer_call_fn_4936
4__inference_batch_normalization_1_layer_call_fn_5037�
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
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1307�
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
annotations� *@�=
;�84������������������������������������
�2�
,__inference_max_pooling2d_layer_call_fn_1313�
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
annotations� *@�=
;�84������������������������������������
�2�
K__inference_average_pooling2d_layer_call_and_return_conditional_losses_1319�
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
annotations� *@�=
;�84������������������������������������
�2�
0__inference_average_pooling2d_layer_call_fn_1325�
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
annotations� *@�=
;�84������������������������������������
�2�
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1331�
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
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_1_layer_call_fn_1337�
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
annotations� *@�=
;�84������������������������������������
�2�
M__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_1343�
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
annotations� *@�=
;�84������������������������������������
�2�
2__inference_average_pooling2d_1_layer_call_fn_1349�
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
annotations� *@�=
;�84������������������������������������
�2�
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1355�
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
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_2_layer_call_fn_1361�
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
annotations� *@�=
;�84������������������������������������
�2�
M__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_1367�
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
annotations� *@�=
;�84������������������������������������
�2�
2__inference_average_pooling2d_2_layer_call_fn_1373�
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
annotations� *@�=
;�84������������������������������������
�2�
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1379�
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
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_3_layer_call_fn_1385�
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
annotations� *@�=
;�84������������������������������������
�2�
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_1391�
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
annotations� *@�=
;�84������������������������������������
�2�
2__inference_average_pooling2d_3_layer_call_fn_1397�
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
annotations� *@�=
;�84������������������������������������
�2�
A__inference_flatten_layer_call_and_return_conditional_losses_5043�
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
&__inference_flatten_layer_call_fn_5048�
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
C__inference_flatten_1_layer_call_and_return_conditional_losses_5054�
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
(__inference_flatten_1_layer_call_fn_5059�
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
C__inference_flatten_2_layer_call_and_return_conditional_losses_5065�
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
(__inference_flatten_2_layer_call_fn_5070�
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
C__inference_flatten_3_layer_call_and_return_conditional_losses_5076�
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
(__inference_flatten_3_layer_call_fn_5081�
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
C__inference_flatten_4_layer_call_and_return_conditional_losses_5087�
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
(__inference_flatten_4_layer_call_fn_5092�
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
C__inference_flatten_5_layer_call_and_return_conditional_losses_5098�
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
(__inference_flatten_5_layer_call_fn_5103�
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
C__inference_flatten_6_layer_call_and_return_conditional_losses_5109�
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
(__inference_flatten_6_layer_call_fn_5114�
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
C__inference_flatten_7_layer_call_and_return_conditional_losses_5120�
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
(__inference_flatten_7_layer_call_fn_5125�
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
E__inference_concatenate_layer_call_and_return_conditional_losses_5138�
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
*__inference_concatenate_layer_call_fn_5150�
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
?__inference_dense_layer_call_and_return_conditional_losses_5172�
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
$__inference_dense_layer_call_fn_5181�
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
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5241
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5273�
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
4__inference_batch_normalization_2_layer_call_fn_5286
4__inference_batch_normalization_2_layer_call_fn_5299�
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
A__inference_p_re_lu_layer_call_and_return_conditional_losses_1598�
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
annotations� *&�#
!�������������������
�2�
&__inference_p_re_lu_layer_call_fn_1606�
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
annotations� *&�#
!�������������������
�2�
A__inference_dropout_layer_call_and_return_conditional_losses_5316
A__inference_dropout_layer_call_and_return_conditional_losses_5311�
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
&__inference_dropout_layer_call_fn_5326
&__inference_dropout_layer_call_fn_5321�
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
A__inference_dense_1_layer_call_and_return_conditional_losses_5337�
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
&__inference_dense_1_layer_call_fn_5346�
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
__inference_loss_fn_0_5357�
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
__inference_loss_fn_1_5368�
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
__inference_loss_fn_2_5379�
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
__inference_loss_fn_3_5390�
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
__inference_loss_fn_4_5401�
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
__inference_loss_fn_5_5412�
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
__inference_loss_fn_6_5423�
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
"__inference_signature_wrapper_3321input_1"�
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
__inference__wrapped_model_997�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������4�1
*�'
%�"
input_1���������
� "1�.
,
dense_1!�
dense_1����������
M__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_1343�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_average_pooling2d_1_layer_call_fn_1349�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_1367�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_average_pooling2d_2_layer_call_fn_1373�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_1391�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_average_pooling2d_3_layer_call_fn_1397�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
K__inference_average_pooling2d_layer_call_and_return_conditional_losses_1319�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_average_pooling2d_layer_call_fn_1325�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4893�VWXYM�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4923�VWXYM�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4981rVWXY;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
0���������
� �
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5011rVWXY;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
0���������
� �
4__inference_batch_normalization_1_layer_call_fn_4936�VWXYM�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
4__inference_batch_normalization_1_layer_call_fn_4949�VWXYM�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
4__inference_batch_normalization_1_layer_call_fn_5024eVWXY;�8
1�.
(�%
inputs���������
p
� " �����������
4__inference_batch_normalization_1_layer_call_fn_5037eVWXY;�8
1�.
(�%
inputs���������
p 
� " �����������
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5241h����4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5273h����4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
4__inference_batch_normalization_2_layer_call_fn_5286[����4�1
*�'
!�
inputs����������
p
� "������������
4__inference_batch_normalization_2_layer_call_fn_5299[����4�1
*�'
!�
inputs����������
p 
� "������������
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4705�MNOPM�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4735�MNOPM�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4793rMNOP;�8
1�.
(�%
inputs���������
p
� "-�*
#� 
0���������
� �
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4823rMNOP;�8
1�.
(�%
inputs���������
p 
� "-�*
#� 
0���������
� �
2__inference_batch_normalization_layer_call_fn_4748�MNOPM�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
2__inference_batch_normalization_layer_call_fn_4761�MNOPM�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
2__inference_batch_normalization_layer_call_fn_4836eMNOP;�8
1�.
(�%
inputs���������
p
� " �����������
2__inference_batch_normalization_layer_call_fn_4849eMNOP;�8
1�.
(�%
inputs���������
p 
� " �����������
E__inference_concatenate_layer_call_and_return_conditional_losses_5138����
���
���
#� 
inputs/0����������
#� 
inputs/1����������
#� 
inputs/2����������
#� 
inputs/3����������
#� 
inputs/4����������
#� 
inputs/5����������
#� 
inputs/6����������
#� 
inputs/7����������
� "&�#
�
0����������
� �
*__inference_concatenate_layer_call_fn_5150����
���
���
#� 
inputs/0����������
#� 
inputs/1����������
#� 
inputs/2����������
#� 
inputs/3����������
#� 
inputs/4����������
#� 
inputs/5����������
#� 
inputs/6����������
#� 
inputs/7����������
� "������������
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4652lFG7�4
-�*
(�%
inputs���������

� "-�*
#� 
0���������
� �
'__inference_conv2d_1_layer_call_fn_4661_FG7�4
-�*
(�%
inputs���������

� " �����������
@__inference_conv2d_layer_call_and_return_conditional_losses_4633l@A7�4
-�*
(�%
inputs���������

� "-�*
#� 
0���������
� �
%__inference_conv2d_layer_call_fn_4642_@A7�4
-�*
(�%
inputs���������

� " �����������
K__inference_custom_activation_layer_call_and_return_conditional_losses_4574�'+36(,47)-58*./90:1;23�0
)�&
$�!
inputs���������
� "[�X
Q�N
%�"
0/0���������

%�"
0/1���������

� �
0__inference_custom_activation_layer_call_fn_4623�'+36(,47)-58*./90:1;23�0
)�&
$�!
inputs���������
� "M�J
#� 
0���������

#� 
1���������
�
A__inference_dense_1_layer_call_and_return_conditional_losses_5337_��0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
&__inference_dense_1_layer_call_fn_5346R��0�-
&�#
!�
inputs����������
� "�����������
?__inference_dense_layer_call_and_return_conditional_losses_5172`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� {
$__inference_dense_layer_call_fn_5181S��0�-
&�#
!�
inputs����������
� "������������
A__inference_dropout_layer_call_and_return_conditional_losses_5311^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
A__inference_dropout_layer_call_and_return_conditional_losses_5316^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� {
&__inference_dropout_layer_call_fn_5321Q4�1
*�'
!�
inputs����������
p
� "�����������{
&__inference_dropout_layer_call_fn_5326Q4�1
*�'
!�
inputs����������
p 
� "������������
C__inference_flatten_1_layer_call_and_return_conditional_losses_5054a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
(__inference_flatten_1_layer_call_fn_5059T7�4
-�*
(�%
inputs���������
� "������������
C__inference_flatten_2_layer_call_and_return_conditional_losses_5065a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
(__inference_flatten_2_layer_call_fn_5070T7�4
-�*
(�%
inputs���������
� "������������
C__inference_flatten_3_layer_call_and_return_conditional_losses_5076a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
(__inference_flatten_3_layer_call_fn_5081T7�4
-�*
(�%
inputs���������
� "������������
C__inference_flatten_4_layer_call_and_return_conditional_losses_5087a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
(__inference_flatten_4_layer_call_fn_5092T7�4
-�*
(�%
inputs���������
� "������������
C__inference_flatten_5_layer_call_and_return_conditional_losses_5098a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
(__inference_flatten_5_layer_call_fn_5103T7�4
-�*
(�%
inputs���������
� "������������
C__inference_flatten_6_layer_call_and_return_conditional_losses_5109a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
(__inference_flatten_6_layer_call_fn_5114T7�4
-�*
(�%
inputs���������
� "������������
C__inference_flatten_7_layer_call_and_return_conditional_losses_5120a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
(__inference_flatten_7_layer_call_fn_5125T7�4
-�*
(�%
inputs���������
� "������������
A__inference_flatten_layer_call_and_return_conditional_losses_5043a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� ~
&__inference_flatten_layer_call_fn_5048T7�4
-�*
(�%
inputs���������
� "�����������9
__inference_loss_fn_0_5357M�

� 
� "� 9
__inference_loss_fn_1_5368N�

� 
� "� 9
__inference_loss_fn_2_5379V�

� 
� "� 9
__inference_loss_fn_3_5390W�

� 
� "� :
__inference_loss_fn_4_5401��

� 
� "� :
__inference_loss_fn_5_5412��

� 
� "� :
__inference_loss_fn_6_5423��

� 
� "� �
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1331�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_1_layer_call_fn_1337�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1355�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_2_layer_call_fn_1361�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1379�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_3_layer_call_fn_1385�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1307�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
,__inference_max_pooling2d_layer_call_fn_1313�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
?__inference_model_layer_call_and_return_conditional_losses_2523�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������<�9
2�/
%�"
input_1���������
p

 
� "%�"
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_2685�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������<�9
2�/
%�"
input_1���������
p 

 
� "%�"
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_3747�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
?__inference_model_layer_call_and_return_conditional_losses_4146�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
$__inference_model_layer_call_fn_2937�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������<�9
2�/
%�"
input_1���������
p

 
� "�����������
$__inference_model_layer_call_fn_3188�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������<�9
2�/
%�"
input_1���������
p 

 
� "�����������
$__inference_model_layer_call_fn_4235�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������;�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_model_layer_call_fn_4324�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������;�8
1�.
$�!
inputs���������
p 

 
� "�����������
A__inference_p_re_lu_layer_call_and_return_conditional_losses_1598f�8�5
.�+
)�&
inputs������������������
� "&�#
�
0����������
� �
&__inference_p_re_lu_layer_call_fn_1606Y�8�5
.�+
)�&
inputs������������������
� "������������
"__inference_signature_wrapper_3321�3'+36(,47)-58*./90:1;2FG@AVWXYMNOP���������?�<
� 
5�2
0
input_1%�"
input_1���������"1�.
,
dense_1!�
dense_1���������