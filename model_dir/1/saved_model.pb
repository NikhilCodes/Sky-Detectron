Ȃ
��
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
dtypetype�
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d388��
�
conv2d_18/kernelVarHandleOp*!
shared_nameconv2d_18/kernel*
dtype0*
_output_shapes
: *
shape:
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*
dtype0*&
_output_shapes
:
t
conv2d_18/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
dtype0*
_output_shapes
:
�
conv2d_19/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:*!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:*
dtype0
t
conv2d_19/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
dtype0*
_output_shapes
:
�
conv2d_20/kernelVarHandleOp*
shape:*!
shared_nameconv2d_20/kernel*
dtype0*
_output_shapes
: 
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*
dtype0*&
_output_shapes
:
t
conv2d_20/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:*
dtype0
�
conv2d_21/kernelVarHandleOp*
shape:*!
shared_nameconv2d_21/kernel*
dtype0*
_output_shapes
: 
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*
dtype0*&
_output_shapes
:
t
conv2d_21/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
:*
dtype0
�
conv2d_22/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: *!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*
dtype0*&
_output_shapes
: 
t
conv2d_22/biasVarHandleOp*
shared_nameconv2d_22/bias*
dtype0*
_output_shapes
: *
shape: 
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
dtype0*
_output_shapes
: 
�
conv2d_23/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:  *!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*
dtype0*&
_output_shapes
:  
t
conv2d_23/biasVarHandleOp*
shared_nameconv2d_23/bias*
dtype0*
_output_shapes
: *
shape: 
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
dtype0*
_output_shapes
: 
�
conv2d_24/kernelVarHandleOp*!
shared_nameconv2d_24/kernel*
dtype0*
_output_shapes
: *
shape:  
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*
dtype0*&
_output_shapes
:  
t
conv2d_24/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
dtype0*
_output_shapes
: 
�
conv2d_25/kernelVarHandleOp*!
shared_nameconv2d_25/kernel*
dtype0*
_output_shapes
: *
shape:  
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_25/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
dtype0*
_output_shapes
: 
�
conv2d_26/kernelVarHandleOp*!
shared_nameconv2d_26/kernel*
dtype0*
_output_shapes
: *
shape: 
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*
dtype0*&
_output_shapes
: 
t
conv2d_26/biasVarHandleOp*
shared_nameconv2d_26/bias*
dtype0*
_output_shapes
: *
shape:
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: *
shape: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2*
dtype0
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
_output_shapes
: *
shape: *
shared_name
Adam/decay*
dtype0
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
�
Adam/conv2d_18/kernel/mVarHandleOp*
shape:*(
shared_nameAdam/conv2d_18/kernel/m*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*
dtype0*&
_output_shapes
:
�
Adam/conv2d_18/bias/mVarHandleOp*&
shared_nameAdam/conv2d_18/bias/m*
dtype0*
_output_shapes
: *
shape:
{
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
dtype0*
_output_shapes
:
�
Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
shape:*(
shared_nameAdam/conv2d_19/kernel/m*
dtype0
�
+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*
dtype0*&
_output_shapes
:
�
Adam/conv2d_19/bias/mVarHandleOp*
shape:*&
shared_nameAdam/conv2d_19/bias/m*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
dtype0*
_output_shapes
:
�
Adam/conv2d_20/kernel/mVarHandleOp*(
shared_nameAdam/conv2d_20/kernel/m*
dtype0*
_output_shapes
: *
shape:
�
+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
shape:*&
shared_nameAdam/conv2d_20/bias/m*
dtype0
{
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
dtype0*
_output_shapes
:
�
Adam/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
shape:*(
shared_nameAdam/conv2d_21/kernel/m*
dtype0
�
+Adam/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/m*
dtype0*&
_output_shapes
:
�
Adam/conv2d_21/bias/mVarHandleOp*
shape:*&
shared_nameAdam/conv2d_21/bias/m*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/m*
dtype0*
_output_shapes
:
�
Adam/conv2d_22/kernel/mVarHandleOp*(
shared_nameAdam/conv2d_22/kernel/m*
dtype0*
_output_shapes
: *
shape: 
�
+Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/m*
dtype0*&
_output_shapes
: 
�
Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
shape: *&
shared_nameAdam/conv2d_22/bias/m*
dtype0
{
)Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/m*
dtype0*
_output_shapes
: 
�
Adam/conv2d_23/kernel/mVarHandleOp*
shape:  *(
shared_nameAdam/conv2d_23/kernel/m*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*
dtype0*&
_output_shapes
:  
�
Adam/conv2d_23/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *&
shared_nameAdam/conv2d_23/bias/m
{
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
dtype0*
_output_shapes
: 
�
Adam/conv2d_24/kernel/mVarHandleOp*
shape:  *(
shared_nameAdam/conv2d_24/kernel/m*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/m*
dtype0*&
_output_shapes
:  
�
Adam/conv2d_24/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *&
shared_nameAdam/conv2d_24/bias/m
{
)Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/m*
dtype0*
_output_shapes
: 
�
Adam/conv2d_25/kernel/mVarHandleOp*(
shared_nameAdam/conv2d_25/kernel/m*
dtype0*
_output_shapes
: *
shape:  
�
+Adam/conv2d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/m*
dtype0*&
_output_shapes
:  
�
Adam/conv2d_25/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *&
shared_nameAdam/conv2d_25/bias/m
{
)Adam/conv2d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/m*
dtype0*
_output_shapes
: 
�
Adam/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
shape: *(
shared_nameAdam/conv2d_26/kernel/m*
dtype0
�
+Adam/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/m*
dtype0*&
_output_shapes
: 
�
Adam/conv2d_26/bias/mVarHandleOp*&
shared_nameAdam/conv2d_26/bias/m*
dtype0*
_output_shapes
: *
shape:
{
)Adam/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/m*
dtype0*
_output_shapes
:
�
Adam/conv2d_18/kernel/vVarHandleOp*(
shared_nameAdam/conv2d_18/kernel/v*
dtype0*
_output_shapes
: *
shape:
�
+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*
dtype0*&
_output_shapes
:
�
Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
shape:*&
shared_nameAdam/conv2d_18/bias/v*
dtype0
{
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
dtype0*
_output_shapes
:
�
Adam/conv2d_19/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*(
shared_nameAdam/conv2d_19/kernel/v
�
+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
shape:*&
shared_nameAdam/conv2d_19/bias/v*
dtype0
{
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
dtype0*
_output_shapes
:
�
Adam/conv2d_20/kernel/vVarHandleOp*
shape:*(
shared_nameAdam/conv2d_20/kernel/v*
dtype0*
_output_shapes
: 
�
+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*
dtype0*&
_output_shapes
:
�
Adam/conv2d_20/bias/vVarHandleOp*
shape:*&
shared_nameAdam/conv2d_20/bias/v*
dtype0*
_output_shapes
: 
{
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
dtype0*
_output_shapes
:
�
Adam/conv2d_21/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*(
shared_nameAdam/conv2d_21/kernel/v
�
+Adam/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/v*
dtype0*&
_output_shapes
:
�
Adam/conv2d_21/bias/vVarHandleOp*&
shared_nameAdam/conv2d_21/bias/v*
dtype0*
_output_shapes
: *
shape:
{
)Adam/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/v*
dtype0*
_output_shapes
:
�
Adam/conv2d_22/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *(
shared_nameAdam/conv2d_22/kernel/v
�
+Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/v*
dtype0*&
_output_shapes
: 
�
Adam/conv2d_22/bias/vVarHandleOp*&
shared_nameAdam/conv2d_22/bias/v*
dtype0*
_output_shapes
: *
shape: 
{
)Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/v*
dtype0*
_output_shapes
: 
�
Adam/conv2d_23/kernel/vVarHandleOp*(
shared_nameAdam/conv2d_23/kernel/v*
dtype0*
_output_shapes
: *
shape:  
�
+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*
dtype0*&
_output_shapes
:  
�
Adam/conv2d_23/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *&
shared_nameAdam/conv2d_23/bias/v
{
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
dtype0*
_output_shapes
: 
�
Adam/conv2d_24/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:  *(
shared_nameAdam/conv2d_24/kernel/v
�
+Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/v*
dtype0*&
_output_shapes
:  
�
Adam/conv2d_24/bias/vVarHandleOp*&
shared_nameAdam/conv2d_24/bias/v*
dtype0*
_output_shapes
: *
shape: 
{
)Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_25/kernel/vVarHandleOp*(
shared_nameAdam/conv2d_25/kernel/v*
dtype0*
_output_shapes
: *
shape:  
�
+Adam/conv2d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/v*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_25/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *&
shared_nameAdam/conv2d_25/bias/v
{
)Adam/conv2d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/v*
dtype0*
_output_shapes
: 
�
Adam/conv2d_26/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *(
shared_nameAdam/conv2d_26/kernel/v
�
+Adam/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/v*
dtype0*&
_output_shapes
: 
�
Adam/conv2d_26/bias/vVarHandleOp*&
shared_nameAdam/conv2d_26/bias/v*
dtype0*
_output_shapes
: *
shape:
{
)Adam/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
�]
ConstConst"/device:CPU:0*�\
value�\B�\ B�\
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
R
.	variables
/trainable_variables
0regularization_losses
1	keras_api
h

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
h

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
h

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
�
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratem�m�m�m�"m�#m�(m�)m�2m�3m�8m�9m�>m�?m�Dm�Em�Jm�Km�v�v�v�v�"v�#v�(v�)v�2v�3v�8v�9v�>v�?v�Dv�Ev�Jv�Kv�
�
0
1
2
3
"4
#5
(6
)7
28
39
810
911
>12
?13
D14
E15
J16
K17
 
�
0
1
2
3
"4
#5
(6
)7
28
39
810
911
>12
?13
D14
E15
J16
K17
�
Umetrics
	variables
regularization_losses
trainable_variables

Vlayers
Wlayer_regularization_losses
Xnon_trainable_variables
 
 
 
 
�
Ymetrics
	variables

Zlayers
trainable_variables
regularization_losses
[layer_regularization_losses
\non_trainable_variables
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
]metrics
	variables

^layers
trainable_variables
regularization_losses
_layer_regularization_losses
`non_trainable_variables
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
ametrics
	variables

blayers
trainable_variables
 regularization_losses
clayer_regularization_losses
dnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
emetrics
$	variables

flayers
%trainable_variables
&regularization_losses
glayer_regularization_losses
hnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
�
imetrics
*	variables

jlayers
+trainable_variables
,regularization_losses
klayer_regularization_losses
lnon_trainable_variables
 
 
 
�
mmetrics
.	variables

nlayers
/trainable_variables
0regularization_losses
olayer_regularization_losses
pnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 
�
qmetrics
4	variables

rlayers
5trainable_variables
6regularization_losses
slayer_regularization_losses
tnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
�
umetrics
:	variables

vlayers
;trainable_variables
<regularization_losses
wlayer_regularization_losses
xnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

>0
?1
 
�
ymetrics
@	variables

zlayers
Atrainable_variables
Bregularization_losses
{layer_regularization_losses
|non_trainable_variables
\Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

D0
E1
 
�
}metrics
F	variables

~layers
Gtrainable_variables
Hregularization_losses
layer_regularization_losses
�non_trainable_variables
\Z
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_26/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
�
�metrics
L	variables
�layers
Mtrainable_variables
Nregularization_losses
 �layer_regularization_losses
�non_trainable_variables
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

�0
F
0
1
2
3
4
5
6
	7

8
9
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


�total

�count
�
_fn_kwargs
�	variables
�trainable_variables
�regularization_losses
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1
 
 
�
�metrics
�	variables
�layers
�trainable_variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
 
 
 

�0
�1
}
VARIABLE_VALUEAdam/conv2d_18/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_21/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_21/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_24/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_24/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_25/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_25/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_18/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_21/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_21/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_24/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_24/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_25/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_25/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
�
serving_default_conv2d_18_inputPlaceholder*1
_output_shapes
:�����������*&
shape:�����������*
dtype0
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_18_inputconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/bias**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*,
_gradient_op_typePartitionedCall-11001*,
f'R%
#__inference_signature_wrapper_10358*
Tout
2
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp+Adam/conv2d_21/kernel/m/Read/ReadVariableOp)Adam/conv2d_21/bias/m/Read/ReadVariableOp+Adam/conv2d_22/kernel/m/Read/ReadVariableOp)Adam/conv2d_22/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp+Adam/conv2d_25/kernel/m/Read/ReadVariableOp)Adam/conv2d_25/bias/m/Read/ReadVariableOp+Adam/conv2d_26/kernel/m/Read/ReadVariableOp)Adam/conv2d_26/bias/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp+Adam/conv2d_21/kernel/v/Read/ReadVariableOp)Adam/conv2d_21/bias/v/Read/ReadVariableOp+Adam/conv2d_22/kernel/v/Read/ReadVariableOp)Adam/conv2d_22/bias/v/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp+Adam/conv2d_25/kernel/v/Read/ReadVariableOp)Adam/conv2d_25/bias/v/Read/ReadVariableOp+Adam/conv2d_26/kernel/v/Read/ReadVariableOp)Adam/conv2d_26/bias/v/Read/ReadVariableOpConst*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *J
TinC
A2?	*,
_gradient_op_typePartitionedCall-11084*'
f"R 
__inference__traced_save_11083
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/conv2d_21/kernel/mAdam/conv2d_21/bias/mAdam/conv2d_22/kernel/mAdam/conv2d_22/bias/mAdam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/conv2d_24/kernel/mAdam/conv2d_24/bias/mAdam/conv2d_25/kernel/mAdam/conv2d_25/bias/mAdam/conv2d_26/kernel/mAdam/conv2d_26/bias/mAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/conv2d_21/kernel/vAdam/conv2d_21/bias/vAdam/conv2d_22/kernel/vAdam/conv2d_22/bias/vAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/vAdam/conv2d_25/kernel/vAdam/conv2d_25/bias/vAdam/conv2d_26/kernel/vAdam/conv2d_26/bias/v*,
_gradient_op_typePartitionedCall-11280**
f%R#
!__inference__traced_restore_11279*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *I
TinB
@2>��
�
�
C__inference_conv2d_20_layer_call_and_return_conditional_losses_9565

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_20/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+���������������������������*
T0*
strides
*
paddingSAME�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+���������������������������*
T0�
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_20/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_20/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_20/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
C__inference_conv2d_23_layer_call_and_return_conditional_losses_9678

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_23/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+��������������������������� �
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� �
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:  *
dtype0�
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_23/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::2h
2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2conv2d_23/kernel/Regularizer/Square/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
(__inference_conv2d_18_layer_call_fn_9512

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+���������������������������*+
_gradient_op_typePartitionedCall-9507*L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_9501*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+���������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
(__inference_conv2d_19_layer_call_fn_9544

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+���������������������������*
Tin
2*+
_gradient_op_typePartitionedCall-9539*L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_9533�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
��
�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9998
conv2d_18_input,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2,
(conv2d_21_statefulpartitionedcall_args_1,
(conv2d_21_statefulpartitionedcall_args_2,
(conv2d_22_statefulpartitionedcall_args_1,
(conv2d_22_statefulpartitionedcall_args_2,
(conv2d_23_statefulpartitionedcall_args_1,
(conv2d_23_statefulpartitionedcall_args_2,
(conv2d_24_statefulpartitionedcall_args_1,
(conv2d_24_statefulpartitionedcall_args_2,
(conv2d_25_statefulpartitionedcall_args_1,
(conv2d_25_statefulpartitionedcall_args_2,
(conv2d_26_statefulpartitionedcall_args_1,
(conv2d_26_statefulpartitionedcall_args_2
identity��!conv2d_18/StatefulPartitionedCall�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�!conv2d_19/StatefulPartitionedCall�2conv2d_19/kernel/Regularizer/Square/ReadVariableOp�!conv2d_20/StatefulPartitionedCall�2conv2d_20/kernel/Regularizer/Square/ReadVariableOp�!conv2d_21/StatefulPartitionedCall�2conv2d_21/kernel/Regularizer/Square/ReadVariableOp�!conv2d_22/StatefulPartitionedCall�2conv2d_22/kernel/Regularizer/Square/ReadVariableOp�!conv2d_23/StatefulPartitionedCall�2conv2d_23/kernel/Regularizer/Square/ReadVariableOp�!conv2d_24/StatefulPartitionedCall�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�!conv2d_25/StatefulPartitionedCall�2conv2d_25/kernel/Regularizer/Square/ReadVariableOp�!conv2d_26/StatefulPartitionedCall�2conv2d_26/kernel/Regularizer/Square/ReadVariableOp�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallconv2d_18_input(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9507*L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_9501*
Tout
2**
config_proto

CPU

GPU 2J 8�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9539*L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_9533*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:������������
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9571*L
fGRE
C__inference_conv2d_20_layer_call_and_return_conditional_losses_9565*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0(conv2d_21_statefulpartitionedcall_args_1(conv2d_21_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9603*L
fGRE
C__inference_conv2d_21_layer_call_and_return_conditional_losses_9597*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:������������
max_pooling2d_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9616*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9622�
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0(conv2d_22_statefulpartitionedcall_args_1(conv2d_22_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9652*L
fGRE
C__inference_conv2d_22_layer_call_and_return_conditional_losses_9646*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:����������� *
Tin
2�
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0(conv2d_23_statefulpartitionedcall_args_1(conv2d_23_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_23_layer_call_and_return_conditional_losses_9678*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:����������� *
Tin
2*+
_gradient_op_typePartitionedCall-9684�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0(conv2d_24_statefulpartitionedcall_args_1(conv2d_24_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_24_layer_call_and_return_conditional_losses_9710*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:����������� *
Tin
2*+
_gradient_op_typePartitionedCall-9716�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0(conv2d_25_statefulpartitionedcall_args_1(conv2d_25_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_25_layer_call_and_return_conditional_losses_9742*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:����������� *+
_gradient_op_typePartitionedCall-9748�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0(conv2d_26_statefulpartitionedcall_args_1(conv2d_26_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2*+
_gradient_op_typePartitionedCall-9780*L
fGRE
C__inference_conv2d_26_layer_call_and_return_conditional_losses_9774*
Tout
2�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_statefulpartitionedcall_args_1"^conv2d_18/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_18/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_18/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_18/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_19_statefulpartitionedcall_args_1"^conv2d_19/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_19/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_19/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_statefulpartitionedcall_args_1"^conv2d_20/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_20/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_statefulpartitionedcall_args_1"^conv2d_21/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_21/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_statefulpartitionedcall_args_1"^conv2d_22/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_22/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_22/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_statefulpartitionedcall_args_1"^conv2d_23/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_23/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_statefulpartitionedcall_args_1"^conv2d_24/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_24/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_25_statefulpartitionedcall_args_1"^conv2d_25/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_25/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_25/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_statefulpartitionedcall_args_1"^conv2d_26/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_26/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp"^conv2d_19/StatefulPartitionedCall3^conv2d_19/kernel/Regularizer/Square/ReadVariableOp"^conv2d_20/StatefulPartitionedCall3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp"^conv2d_21/StatefulPartitionedCall3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp"^conv2d_22/StatefulPartitionedCall3^conv2d_22/kernel/Regularizer/Square/ReadVariableOp"^conv2d_23/StatefulPartitionedCall3^conv2d_23/kernel/Regularizer/Square/ReadVariableOp"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp"^conv2d_25/StatefulPartitionedCall3^conv2d_25/kernel/Regularizer/Square/ReadVariableOp"^conv2d_26/StatefulPartitionedCall3^conv2d_26/kernel/Regularizer/Square/ReadVariableOp*1
_output_shapes
:�����������*
T0"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2h
2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2h
2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2conv2d_19/kernel/Regularizer/Square/ReadVariableOp:	 :
 : : : : : : : : :/ +
)
_user_specified_nameconv2d_18_input: : : : : : : : 
�
�
C__inference_conv2d_24_layer_call_and_return_conditional_losses_9710

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*A
_output_shapes/
-:+��������������������������� *
T0*
strides
�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� �
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:  *
dtype0�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_24/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9616

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
(__inference_conv2d_25_layer_call_fn_9753

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+��������������������������� *
Tin
2*+
_gradient_op_typePartitionedCall-9748*L
fGRE
C__inference_conv2d_25_layer_call_and_return_conditional_losses_9742*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
(__inference_conv2d_24_layer_call_fn_9721

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+��������������������������� *+
_gradient_op_typePartitionedCall-9716*L
fGRE
C__inference_conv2d_24_layer_call_and_return_conditional_losses_9710*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
C__inference_conv2d_26_layer_call_and_return_conditional_losses_9774

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_26/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+����������������������������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+����������������������������
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_26/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_26/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_26/kernel/Regularizer/Square/ReadVariableOp*A
_output_shapes/
-:+���������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::2h
2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2conv2d_26/kernel/Regularizer/Square/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
__inference_loss_fn_3_10798?
;conv2d_21_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_21/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_21_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentity$conv2d_21/kernel/Regularizer/add:z:03^conv2d_21/kernel/Regularizer/Square/ReadVariableOp*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp:  
�
�
__inference_loss_fn_2_10783?
;conv2d_20_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_20/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_20_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_20/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_20/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentity$conv2d_20/kernel/Regularizer/add:z:03^conv2d_20/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp:  
��
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_10235

inputs,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2,
(conv2d_21_statefulpartitionedcall_args_1,
(conv2d_21_statefulpartitionedcall_args_2,
(conv2d_22_statefulpartitionedcall_args_1,
(conv2d_22_statefulpartitionedcall_args_2,
(conv2d_23_statefulpartitionedcall_args_1,
(conv2d_23_statefulpartitionedcall_args_2,
(conv2d_24_statefulpartitionedcall_args_1,
(conv2d_24_statefulpartitionedcall_args_2,
(conv2d_25_statefulpartitionedcall_args_1,
(conv2d_25_statefulpartitionedcall_args_2,
(conv2d_26_statefulpartitionedcall_args_1,
(conv2d_26_statefulpartitionedcall_args_2
identity��!conv2d_18/StatefulPartitionedCall�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�!conv2d_19/StatefulPartitionedCall�2conv2d_19/kernel/Regularizer/Square/ReadVariableOp�!conv2d_20/StatefulPartitionedCall�2conv2d_20/kernel/Regularizer/Square/ReadVariableOp�!conv2d_21/StatefulPartitionedCall�2conv2d_21/kernel/Regularizer/Square/ReadVariableOp�!conv2d_22/StatefulPartitionedCall�2conv2d_22/kernel/Regularizer/Square/ReadVariableOp�!conv2d_23/StatefulPartitionedCall�2conv2d_23/kernel/Regularizer/Square/ReadVariableOp�!conv2d_24/StatefulPartitionedCall�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�!conv2d_25/StatefulPartitionedCall�2conv2d_25/kernel/Regularizer/Square/ReadVariableOp�!conv2d_26/StatefulPartitionedCall�2conv2d_26/kernel/Regularizer/Square/ReadVariableOp�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9507*L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_9501*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9539*L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_9533*
Tout
2�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9571*L
fGRE
C__inference_conv2d_20_layer_call_and_return_conditional_losses_9565*
Tout
2�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0(conv2d_21_statefulpartitionedcall_args_1(conv2d_21_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9603*L
fGRE
C__inference_conv2d_21_layer_call_and_return_conditional_losses_9597*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2�
max_pooling2d_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-9622*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9616*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2�
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0(conv2d_22_statefulpartitionedcall_args_1(conv2d_22_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9652*L
fGRE
C__inference_conv2d_22_layer_call_and_return_conditional_losses_9646*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:����������� �
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0(conv2d_23_statefulpartitionedcall_args_1(conv2d_23_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_23_layer_call_and_return_conditional_losses_9678*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:����������� *
Tin
2*+
_gradient_op_typePartitionedCall-9684�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0(conv2d_24_statefulpartitionedcall_args_1(conv2d_24_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:����������� *+
_gradient_op_typePartitionedCall-9716*L
fGRE
C__inference_conv2d_24_layer_call_and_return_conditional_losses_9710*
Tout
2�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0(conv2d_25_statefulpartitionedcall_args_1(conv2d_25_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:����������� *+
_gradient_op_typePartitionedCall-9748*L
fGRE
C__inference_conv2d_25_layer_call_and_return_conditional_losses_9742*
Tout
2�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0(conv2d_26_statefulpartitionedcall_args_1(conv2d_26_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2*+
_gradient_op_typePartitionedCall-9780*L
fGRE
C__inference_conv2d_26_layer_call_and_return_conditional_losses_9774�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_statefulpartitionedcall_args_1"^conv2d_18/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_18/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_18/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_18/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_19_statefulpartitionedcall_args_1"^conv2d_19/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_19/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_19/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_19/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_statefulpartitionedcall_args_1"^conv2d_20/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_20/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_statefulpartitionedcall_args_1"^conv2d_21/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_21/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_statefulpartitionedcall_args_1"^conv2d_22/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_22/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_22/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_statefulpartitionedcall_args_1"^conv2d_23/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_23/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_23/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_statefulpartitionedcall_args_1"^conv2d_24/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_24/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_25_statefulpartitionedcall_args_1"^conv2d_25/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_25/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_25/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_statefulpartitionedcall_args_1"^conv2d_26/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
: *
T0{
"conv2d_26/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_26/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    �
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp"^conv2d_19/StatefulPartitionedCall3^conv2d_19/kernel/Regularizer/Square/ReadVariableOp"^conv2d_20/StatefulPartitionedCall3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp"^conv2d_21/StatefulPartitionedCall3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp"^conv2d_22/StatefulPartitionedCall3^conv2d_22/kernel/Regularizer/Square/ReadVariableOp"^conv2d_23/StatefulPartitionedCall3^conv2d_23/kernel/Regularizer/Square/ReadVariableOp"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp"^conv2d_25/StatefulPartitionedCall3^conv2d_25/kernel/Regularizer/Square/ReadVariableOp"^conv2d_26/StatefulPartitionedCall3^conv2d_26/kernel/Regularizer/Square/ReadVariableOp*1
_output_shapes
:�����������*
T0"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2h
2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2h
2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp: : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : 
�
�
,__inference_sequential_2_layer_call_fn_10127
conv2d_18_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_18_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*,
_gradient_op_typePartitionedCall-10106*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_10105*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_18_input: : : : : : : : :	 :
 : : : : : : : : 
�
�
C__inference_conv2d_21_layer_call_and_return_conditional_losses_9597

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_21/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+����������������������������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+����������������������������
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_21/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp*A
_output_shapes/
-:+���������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ط
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_10622

inputs,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource,
(conv2d_21_conv2d_readvariableop_resource-
)conv2d_21_biasadd_readvariableop_resource,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource
identity�� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�2conv2d_19/kernel/Regularizer/Square/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�2conv2d_20/kernel/Regularizer/Square/ReadVariableOp� conv2d_21/BiasAdd/ReadVariableOp�conv2d_21/Conv2D/ReadVariableOp�2conv2d_21/kernel/Regularizer/Square/ReadVariableOp� conv2d_22/BiasAdd/ReadVariableOp�conv2d_22/Conv2D/ReadVariableOp�2conv2d_22/kernel/Regularizer/Square/ReadVariableOp� conv2d_23/BiasAdd/ReadVariableOp�conv2d_23/Conv2D/ReadVariableOp�2conv2d_23/kernel/Regularizer/Square/ReadVariableOp� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp�2conv2d_25/kernel/Regularizer/Square/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp�2conv2d_26/kernel/Regularizer/Square/ReadVariableOp�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:�����������*
T0*
strides
�
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
conv2d_19/Conv2DConv2Dconv2d_18/BiasAdd:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:�����������*
T0*
strides
�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
conv2d_20/Conv2DConv2Dconv2d_19/BiasAdd:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:�����������*
T0*
strides
�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
conv2d_21/Conv2DConv2Dconv2d_20/BiasAdd:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:������������
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0�
max_pooling2d_2/MaxPoolMaxPoolconv2d_21/BiasAdd:output:0*1
_output_shapes
:�����������*
strides
*
ksize
*
paddingVALID�
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
conv2d_22/Conv2DConv2D max_pooling2d_2/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0*
strides
*
paddingSAME�
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
conv2d_23/Conv2DConv2Dconv2d_22/BiasAdd:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:����������� *
T0*
strides
�
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:  *
dtype0�
conv2d_24/Conv2DConv2Dconv2d_23/BiasAdd:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:����������� *
T0*
strides
�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0�
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
conv2d_25/Conv2DConv2Dconv2d_24/BiasAdd:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:����������� *
T0*
strides
�
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0�
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
conv2d_26/Conv2DConv2Dconv2d_25/BiasAdd:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:������������
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource ^conv2d_18/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_18/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_18/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_18/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource ^conv2d_19/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource ^conv2d_20/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_20/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_20/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_20/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource ^conv2d_21/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_21/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_21/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource ^conv2d_22/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_22/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_22/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_22/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource ^conv2d_23/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_23/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource ^conv2d_24/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_24/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource ^conv2d_25/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_25/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource ^conv2d_26/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0�
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
: *
T0{
"conv2d_26/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_26/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_26/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�	
IdentityIdentityconv2d_26/BiasAdd:output:0!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp3^conv2d_19/kernel/Regularizer/Square/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp3^conv2d_22/kernel/Regularizer/Square/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp3^conv2d_23/kernel/Regularizer/Square/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp3^conv2d_25/kernel/Regularizer/Square/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp3^conv2d_26/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2h
2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2h
2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2h
2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : 
�
J
.__inference_max_pooling2d_2_layer_call_fn_9625

inputs
identity�
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-9622*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9616*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4�������������������������������������
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4������������������������������������*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_10753?
;conv2d_18_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_18_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_18/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_18/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_18/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentity$conv2d_18/kernel/Regularizer/add:z:03^conv2d_18/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp:  
�
�
(__inference_conv2d_22_layer_call_fn_9657

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+��������������������������� *+
_gradient_op_typePartitionedCall-9652*L
fGRE
C__inference_conv2d_22_layer_call_and_return_conditional_losses_9646*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
__inference_loss_fn_5_10828?
;conv2d_23_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_23/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_23_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_23/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_23/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    �
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentity$conv2d_23/kernel/Regularizer/add:z:03^conv2d_23/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2conv2d_23/kernel/Regularizer/Square/ReadVariableOp:  
�
�
__inference_loss_fn_4_10813?
;conv2d_22_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_22/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_22_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_22/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_22/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentity$conv2d_22/kernel/Regularizer/add:z:03^conv2d_22/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2conv2d_22/kernel/Regularizer/Square/ReadVariableOp:  
ط
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_10491

inputs,
(conv2d_18_conv2d_readvariableop_resource-
)conv2d_18_biasadd_readvariableop_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource,
(conv2d_21_conv2d_readvariableop_resource-
)conv2d_21_biasadd_readvariableop_resource,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource
identity�� conv2d_18/BiasAdd/ReadVariableOp�conv2d_18/Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp� conv2d_19/BiasAdd/ReadVariableOp�conv2d_19/Conv2D/ReadVariableOp�2conv2d_19/kernel/Regularizer/Square/ReadVariableOp� conv2d_20/BiasAdd/ReadVariableOp�conv2d_20/Conv2D/ReadVariableOp�2conv2d_20/kernel/Regularizer/Square/ReadVariableOp� conv2d_21/BiasAdd/ReadVariableOp�conv2d_21/Conv2D/ReadVariableOp�2conv2d_21/kernel/Regularizer/Square/ReadVariableOp� conv2d_22/BiasAdd/ReadVariableOp�conv2d_22/Conv2D/ReadVariableOp�2conv2d_22/kernel/Regularizer/Square/ReadVariableOp� conv2d_23/BiasAdd/ReadVariableOp�conv2d_23/Conv2D/ReadVariableOp�2conv2d_23/kernel/Regularizer/Square/ReadVariableOp� conv2d_24/BiasAdd/ReadVariableOp�conv2d_24/Conv2D/ReadVariableOp�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp� conv2d_25/BiasAdd/ReadVariableOp�conv2d_25/Conv2D/ReadVariableOp�2conv2d_25/kernel/Regularizer/Square/ReadVariableOp� conv2d_26/BiasAdd/ReadVariableOp�conv2d_26/Conv2D/ReadVariableOp�2conv2d_26/kernel/Regularizer/Square/ReadVariableOp�
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:������������
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
conv2d_19/Conv2DConv2Dconv2d_18/BiasAdd:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*1
_output_shapes
:�����������*
T0�
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0�
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0�
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
conv2d_20/Conv2DConv2Dconv2d_19/BiasAdd:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0*
strides
*
paddingSAME�
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0�
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
conv2d_21/Conv2DConv2Dconv2d_20/BiasAdd:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0*
strides
*
paddingSAME�
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
max_pooling2d_2/MaxPoolMaxPoolconv2d_21/BiasAdd:output:0*
ksize
*
paddingVALID*1
_output_shapes
:�����������*
strides
�
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0�
conv2d_22/Conv2DConv2D max_pooling2d_2/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:����������� *
T0*
strides
�
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
conv2d_23/Conv2DConv2Dconv2d_22/BiasAdd:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0*
strides
*
paddingSAME�
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0�
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
conv2d_24/Conv2DConv2Dconv2d_23/BiasAdd:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:����������� *
T0*
strides
�
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0�
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
conv2d_25/Conv2DConv2Dconv2d_24/BiasAdd:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0*
strides
*
paddingSAME�
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
conv2d_26/Conv2DConv2Dconv2d_25/BiasAdd:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:������������
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource ^conv2d_18/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_18/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_18/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource ^conv2d_19/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_19/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource ^conv2d_20/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_20/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource ^conv2d_21/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_21/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_21/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource ^conv2d_22/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
: *
T0{
"conv2d_22/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_22/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_22/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource ^conv2d_23/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_23/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_23/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource ^conv2d_24/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:  *
dtype0�
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_24/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource ^conv2d_25/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_25/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_25/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource ^conv2d_26/Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    �
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�	
IdentityIdentityconv2d_26/BiasAdd:output:0!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp3^conv2d_19/kernel/Regularizer/Square/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp3^conv2d_22/kernel/Regularizer/Square/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp3^conv2d_23/kernel/Regularizer/Square/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp3^conv2d_25/kernel/Regularizer/Square/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp3^conv2d_26/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2h
2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2h
2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2h
2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2h
2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2conv2d_25/kernel/Regularizer/Square/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : 
�
�
(__inference_conv2d_23_layer_call_fn_9689

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9684*L
fGRE
C__inference_conv2d_23_layer_call_and_return_conditional_losses_9678*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+��������������������������� *
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
(__inference_conv2d_21_layer_call_fn_9608

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+���������������������������*+
_gradient_op_typePartitionedCall-9603*L
fGRE
C__inference_conv2d_21_layer_call_and_return_conditional_losses_9597*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
#__inference_signature_wrapper_10358
conv2d_18_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_18_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2*,
_gradient_op_typePartitionedCall-10337*(
f#R!
__inference__wrapped_model_9480�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*1
_output_shapes
:�����������*
T0"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :/ +
)
_user_specified_nameconv2d_18_input: : : : : : : : :	 :
 : 
�n
�
__inference__traced_save_11083
file_prefix/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop6
2savev2_adam_conv2d_21_kernel_m_read_readvariableop4
0savev2_adam_conv2d_21_bias_m_read_readvariableop6
2savev2_adam_conv2d_22_kernel_m_read_readvariableop4
0savev2_adam_conv2d_22_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableop6
2savev2_adam_conv2d_25_kernel_m_read_readvariableop4
0savev2_adam_conv2d_25_bias_m_read_readvariableop6
2savev2_adam_conv2d_26_kernel_m_read_readvariableop4
0savev2_adam_conv2d_26_bias_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop6
2savev2_adam_conv2d_21_kernel_v_read_readvariableop4
0savev2_adam_conv2d_21_bias_v_read_readvariableop6
2savev2_adam_conv2d_22_kernel_v_read_readvariableop4
0savev2_adam_conv2d_22_bias_v_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableop6
2savev2_adam_conv2d_25_kernel_v_read_readvariableop4
0savev2_adam_conv2d_25_bias_v_read_readvariableop6
2savev2_adam_conv2d_26_kernel_v_read_readvariableop4
0savev2_adam_conv2d_26_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_b2b0586bb13048239faf95d544212b18/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �"
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:=*�!
value�!B�!=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE�
SaveV2/shape_and_slicesConst"/device:CPU:0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:=�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop2savev2_adam_conv2d_21_kernel_m_read_readvariableop0savev2_adam_conv2d_21_bias_m_read_readvariableop2savev2_adam_conv2d_22_kernel_m_read_readvariableop0savev2_adam_conv2d_22_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop2savev2_adam_conv2d_25_kernel_m_read_readvariableop0savev2_adam_conv2d_25_bias_m_read_readvariableop2savev2_adam_conv2d_26_kernel_m_read_readvariableop0savev2_adam_conv2d_26_bias_m_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop2savev2_adam_conv2d_21_kernel_v_read_readvariableop0savev2_adam_conv2d_21_bias_v_read_readvariableop2savev2_adam_conv2d_22_kernel_v_read_readvariableop0savev2_adam_conv2d_22_bias_v_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop2savev2_adam_conv2d_25_kernel_v_read_readvariableop0savev2_adam_conv2d_25_bias_v_read_readvariableop2savev2_adam_conv2d_26_kernel_v_read_readvariableop0savev2_adam_conv2d_26_bias_v_read_readvariableop"/device:CPU:0*K
dtypesA
?2=	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::: : :  : :  : :  : : :: : : : : : : ::::::::: : :  : :  : :  : : :::::::::: : :  : :  : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> 
�
�
,__inference_sequential_2_layer_call_fn_10257
conv2d_18_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_18_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2*,
_gradient_op_typePartitionedCall-10236*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_10235*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*1
_output_shapes
:�����������*
T0"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_18_input: : : : : : : : :	 :
 : : : : : : : : 
�
�
C__inference_conv2d_25_layer_call_and_return_conditional_losses_9742

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_25/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+��������������������������� *
T0*
strides
*
paddingSAME�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� �
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_25/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_25/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2conv2d_25/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
C__inference_conv2d_19_layer_call_and_return_conditional_losses_9533

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_19/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*A
_output_shapes/
-:+����������������������������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+����������������������������
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_19/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_19/kernel/Regularizer/Square/ReadVariableOp*A
_output_shapes/
-:+���������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2h
2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2conv2d_19/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
(__inference_conv2d_26_layer_call_fn_9785

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+���������������������������*+
_gradient_op_typePartitionedCall-9780*L
fGRE
C__inference_conv2d_26_layer_call_and_return_conditional_losses_9774*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+���������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
C__inference_conv2d_18_layer_call_and_return_conditional_losses_9501

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*A
_output_shapes/
-:+���������������������������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+����������������������������
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_18/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_18/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_18/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp*A
_output_shapes/
-:+���������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
__inference_loss_fn_6_10843?
;conv2d_24_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_24_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_24/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentity$conv2d_24/kernel/Regularizer/add:z:03^conv2d_24/kernel/Regularizer/Square/ReadVariableOp*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp:  
�
�
,__inference_sequential_2_layer_call_fn_10668

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*1
_output_shapes
:�����������*
Tin
2*,
_gradient_op_typePartitionedCall-10236*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_10235*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:
 : : : : : : : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 
�
�
__inference_loss_fn_1_10768?
;conv2d_19_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_19/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_19_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_19/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentity$conv2d_19/kernel/Regularizer/add:z:03^conv2d_19/kernel/Regularizer/Square/ReadVariableOp*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2conv2d_19/kernel/Regularizer/Square/ReadVariableOp:  
��
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_10105

inputs,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2,
(conv2d_21_statefulpartitionedcall_args_1,
(conv2d_21_statefulpartitionedcall_args_2,
(conv2d_22_statefulpartitionedcall_args_1,
(conv2d_22_statefulpartitionedcall_args_2,
(conv2d_23_statefulpartitionedcall_args_1,
(conv2d_23_statefulpartitionedcall_args_2,
(conv2d_24_statefulpartitionedcall_args_1,
(conv2d_24_statefulpartitionedcall_args_2,
(conv2d_25_statefulpartitionedcall_args_1,
(conv2d_25_statefulpartitionedcall_args_2,
(conv2d_26_statefulpartitionedcall_args_1,
(conv2d_26_statefulpartitionedcall_args_2
identity��!conv2d_18/StatefulPartitionedCall�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�!conv2d_19/StatefulPartitionedCall�2conv2d_19/kernel/Regularizer/Square/ReadVariableOp�!conv2d_20/StatefulPartitionedCall�2conv2d_20/kernel/Regularizer/Square/ReadVariableOp�!conv2d_21/StatefulPartitionedCall�2conv2d_21/kernel/Regularizer/Square/ReadVariableOp�!conv2d_22/StatefulPartitionedCall�2conv2d_22/kernel/Regularizer/Square/ReadVariableOp�!conv2d_23/StatefulPartitionedCall�2conv2d_23/kernel/Regularizer/Square/ReadVariableOp�!conv2d_24/StatefulPartitionedCall�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�!conv2d_25/StatefulPartitionedCall�2conv2d_25/kernel/Regularizer/Square/ReadVariableOp�!conv2d_26/StatefulPartitionedCall�2conv2d_26/kernel/Regularizer/Square/ReadVariableOp�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9507*L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_9501*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:������������
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9539*L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_9533*
Tout
2�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9571*L
fGRE
C__inference_conv2d_20_layer_call_and_return_conditional_losses_9565*
Tout
2�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0(conv2d_21_statefulpartitionedcall_args_1(conv2d_21_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2*+
_gradient_op_typePartitionedCall-9603*L
fGRE
C__inference_conv2d_21_layer_call_and_return_conditional_losses_9597*
Tout
2�
max_pooling2d_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-9622*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9616*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:������������
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0(conv2d_22_statefulpartitionedcall_args_1(conv2d_22_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:����������� *
Tin
2*+
_gradient_op_typePartitionedCall-9652*L
fGRE
C__inference_conv2d_22_layer_call_and_return_conditional_losses_9646*
Tout
2�
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0(conv2d_23_statefulpartitionedcall_args_1(conv2d_23_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:����������� *+
_gradient_op_typePartitionedCall-9684*L
fGRE
C__inference_conv2d_23_layer_call_and_return_conditional_losses_9678*
Tout
2�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0(conv2d_24_statefulpartitionedcall_args_1(conv2d_24_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:����������� *
Tin
2*+
_gradient_op_typePartitionedCall-9716*L
fGRE
C__inference_conv2d_24_layer_call_and_return_conditional_losses_9710*
Tout
2�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0(conv2d_25_statefulpartitionedcall_args_1(conv2d_25_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:����������� *+
_gradient_op_typePartitionedCall-9748*L
fGRE
C__inference_conv2d_25_layer_call_and_return_conditional_losses_9742*
Tout
2�
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0(conv2d_26_statefulpartitionedcall_args_1(conv2d_26_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9780*L
fGRE
C__inference_conv2d_26_layer_call_and_return_conditional_losses_9774*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_statefulpartitionedcall_args_1"^conv2d_18/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_18/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_18/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_18/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    �
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_19_statefulpartitionedcall_args_1"^conv2d_19/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_19/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_19/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_19/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    �
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_statefulpartitionedcall_args_1"^conv2d_20/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_20/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_statefulpartitionedcall_args_1"^conv2d_21/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_21/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_statefulpartitionedcall_args_1"^conv2d_22/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_22/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_22/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_22/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_statefulpartitionedcall_args_1"^conv2d_23/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_23/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_23/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_statefulpartitionedcall_args_1"^conv2d_24/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_24/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_24/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    �
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_25_statefulpartitionedcall_args_1"^conv2d_25/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_25/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_25/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_statefulpartitionedcall_args_1"^conv2d_26/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_26/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp"^conv2d_19/StatefulPartitionedCall3^conv2d_19/kernel/Regularizer/Square/ReadVariableOp"^conv2d_20/StatefulPartitionedCall3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp"^conv2d_21/StatefulPartitionedCall3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp"^conv2d_22/StatefulPartitionedCall3^conv2d_22/kernel/Regularizer/Square/ReadVariableOp"^conv2d_23/StatefulPartitionedCall3^conv2d_23/kernel/Regularizer/Square/ReadVariableOp"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp"^conv2d_25/StatefulPartitionedCall3^conv2d_25/kernel/Regularizer/Square/ReadVariableOp"^conv2d_26/StatefulPartitionedCall3^conv2d_26/kernel/Regularizer/Square/ReadVariableOp*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2h
2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2h
2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp:	 :
 : : : : : : : : :& "
 
_user_specified_nameinputs: : : : : : : : 
�
�
(__inference_conv2d_20_layer_call_fn_9576

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+���������������������������*
Tin
2*+
_gradient_op_typePartitionedCall-9571*L
fGRE
C__inference_conv2d_20_layer_call_and_return_conditional_losses_9565*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�e
�
__inference__wrapped_model_9480
conv2d_18_input9
5sequential_2_conv2d_18_conv2d_readvariableop_resource:
6sequential_2_conv2d_18_biasadd_readvariableop_resource9
5sequential_2_conv2d_19_conv2d_readvariableop_resource:
6sequential_2_conv2d_19_biasadd_readvariableop_resource9
5sequential_2_conv2d_20_conv2d_readvariableop_resource:
6sequential_2_conv2d_20_biasadd_readvariableop_resource9
5sequential_2_conv2d_21_conv2d_readvariableop_resource:
6sequential_2_conv2d_21_biasadd_readvariableop_resource9
5sequential_2_conv2d_22_conv2d_readvariableop_resource:
6sequential_2_conv2d_22_biasadd_readvariableop_resource9
5sequential_2_conv2d_23_conv2d_readvariableop_resource:
6sequential_2_conv2d_23_biasadd_readvariableop_resource9
5sequential_2_conv2d_24_conv2d_readvariableop_resource:
6sequential_2_conv2d_24_biasadd_readvariableop_resource9
5sequential_2_conv2d_25_conv2d_readvariableop_resource:
6sequential_2_conv2d_25_biasadd_readvariableop_resource9
5sequential_2_conv2d_26_conv2d_readvariableop_resource:
6sequential_2_conv2d_26_biasadd_readvariableop_resource
identity��-sequential_2/conv2d_18/BiasAdd/ReadVariableOp�,sequential_2/conv2d_18/Conv2D/ReadVariableOp�-sequential_2/conv2d_19/BiasAdd/ReadVariableOp�,sequential_2/conv2d_19/Conv2D/ReadVariableOp�-sequential_2/conv2d_20/BiasAdd/ReadVariableOp�,sequential_2/conv2d_20/Conv2D/ReadVariableOp�-sequential_2/conv2d_21/BiasAdd/ReadVariableOp�,sequential_2/conv2d_21/Conv2D/ReadVariableOp�-sequential_2/conv2d_22/BiasAdd/ReadVariableOp�,sequential_2/conv2d_22/Conv2D/ReadVariableOp�-sequential_2/conv2d_23/BiasAdd/ReadVariableOp�,sequential_2/conv2d_23/Conv2D/ReadVariableOp�-sequential_2/conv2d_24/BiasAdd/ReadVariableOp�,sequential_2/conv2d_24/Conv2D/ReadVariableOp�-sequential_2/conv2d_25/BiasAdd/ReadVariableOp�,sequential_2/conv2d_25/Conv2D/ReadVariableOp�-sequential_2/conv2d_26/BiasAdd/ReadVariableOp�,sequential_2/conv2d_26/Conv2D/ReadVariableOp�
,sequential_2/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_18_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
sequential_2/conv2d_18/Conv2DConv2Dconv2d_18_input4sequential_2/conv2d_18/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:�����������*
T0*
strides
�
-sequential_2/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_18_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
sequential_2/conv2d_18/BiasAddBiasAdd&sequential_2/conv2d_18/Conv2D:output:05sequential_2/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
,sequential_2/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_19_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
sequential_2/conv2d_19/Conv2DConv2D'sequential_2/conv2d_18/BiasAdd:output:04sequential_2/conv2d_19/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0*
strides
*
paddingSAME�
-sequential_2/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_19_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
sequential_2/conv2d_19/BiasAddBiasAdd&sequential_2/conv2d_19/Conv2D:output:05sequential_2/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
,sequential_2/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_20_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
sequential_2/conv2d_20/Conv2DConv2D'sequential_2/conv2d_19/BiasAdd:output:04sequential_2/conv2d_20/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:�����������*
T0*
strides
�
-sequential_2/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_20_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
sequential_2/conv2d_20/BiasAddBiasAdd&sequential_2/conv2d_20/Conv2D:output:05sequential_2/conv2d_20/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0�
,sequential_2/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_21_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
sequential_2/conv2d_21/Conv2DConv2D'sequential_2/conv2d_20/BiasAdd:output:04sequential_2/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:������������
-sequential_2/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_21_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
sequential_2/conv2d_21/BiasAddBiasAdd&sequential_2/conv2d_21/Conv2D:output:05sequential_2/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
$sequential_2/max_pooling2d_2/MaxPoolMaxPool'sequential_2/conv2d_21/BiasAdd:output:0*1
_output_shapes
:�����������*
strides
*
ksize
*
paddingVALID�
,sequential_2/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_22_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
sequential_2/conv2d_22/Conv2DConv2D-sequential_2/max_pooling2d_2/MaxPool:output:04sequential_2/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:����������� �
-sequential_2/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_22_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
sequential_2/conv2d_22/BiasAddBiasAdd&sequential_2/conv2d_22/Conv2D:output:05sequential_2/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
,sequential_2/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_23_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
sequential_2/conv2d_23/Conv2DConv2D'sequential_2/conv2d_22/BiasAdd:output:04sequential_2/conv2d_23/Conv2D/ReadVariableOp:value:0*
paddingSAME*1
_output_shapes
:����������� *
T0*
strides
�
-sequential_2/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_23_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0�
sequential_2/conv2d_23/BiasAddBiasAdd&sequential_2/conv2d_23/Conv2D:output:05sequential_2/conv2d_23/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0�
,sequential_2/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_24_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
sequential_2/conv2d_24/Conv2DConv2D'sequential_2/conv2d_23/BiasAdd:output:04sequential_2/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*1
_output_shapes
:����������� �
-sequential_2/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_24_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
sequential_2/conv2d_24/BiasAddBiasAdd&sequential_2/conv2d_24/Conv2D:output:05sequential_2/conv2d_24/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0�
,sequential_2/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_25_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
sequential_2/conv2d_25/Conv2DConv2D'sequential_2/conv2d_24/BiasAdd:output:04sequential_2/conv2d_25/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0*
strides
*
paddingSAME�
-sequential_2/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_25_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
sequential_2/conv2d_25/BiasAddBiasAdd&sequential_2/conv2d_25/Conv2D:output:05sequential_2/conv2d_25/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:����������� *
T0�
,sequential_2/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_26_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
sequential_2/conv2d_26/Conv2DConv2D'sequential_2/conv2d_25/BiasAdd:output:04sequential_2/conv2d_26/Conv2D/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0*
strides
*
paddingSAME�
-sequential_2/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_26_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
sequential_2/conv2d_26/BiasAddBiasAdd&sequential_2/conv2d_26/Conv2D:output:05sequential_2/conv2d_26/BiasAdd/ReadVariableOp:value:0*1
_output_shapes
:�����������*
T0�
IdentityIdentity'sequential_2/conv2d_26/BiasAdd:output:0.^sequential_2/conv2d_18/BiasAdd/ReadVariableOp-^sequential_2/conv2d_18/Conv2D/ReadVariableOp.^sequential_2/conv2d_19/BiasAdd/ReadVariableOp-^sequential_2/conv2d_19/Conv2D/ReadVariableOp.^sequential_2/conv2d_20/BiasAdd/ReadVariableOp-^sequential_2/conv2d_20/Conv2D/ReadVariableOp.^sequential_2/conv2d_21/BiasAdd/ReadVariableOp-^sequential_2/conv2d_21/Conv2D/ReadVariableOp.^sequential_2/conv2d_22/BiasAdd/ReadVariableOp-^sequential_2/conv2d_22/Conv2D/ReadVariableOp.^sequential_2/conv2d_23/BiasAdd/ReadVariableOp-^sequential_2/conv2d_23/Conv2D/ReadVariableOp.^sequential_2/conv2d_24/BiasAdd/ReadVariableOp-^sequential_2/conv2d_24/Conv2D/ReadVariableOp.^sequential_2/conv2d_25/BiasAdd/ReadVariableOp-^sequential_2/conv2d_25/Conv2D/ReadVariableOp.^sequential_2/conv2d_26/BiasAdd/ReadVariableOp-^sequential_2/conv2d_26/Conv2D/ReadVariableOp*1
_output_shapes
:�����������*
T0"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2^
-sequential_2/conv2d_22/BiasAdd/ReadVariableOp-sequential_2/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_26/Conv2D/ReadVariableOp,sequential_2/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_23/Conv2D/ReadVariableOp,sequential_2/conv2d_23/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_18/Conv2D/ReadVariableOp,sequential_2/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_20/BiasAdd/ReadVariableOp-sequential_2/conv2d_20/BiasAdd/ReadVariableOp2^
-sequential_2/conv2d_25/BiasAdd/ReadVariableOp-sequential_2/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_20/Conv2D/ReadVariableOp,sequential_2/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_18/BiasAdd/ReadVariableOp-sequential_2/conv2d_18/BiasAdd/ReadVariableOp2^
-sequential_2/conv2d_23/BiasAdd/ReadVariableOp-sequential_2/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_24/Conv2D/ReadVariableOp,sequential_2/conv2d_24/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_19/Conv2D/ReadVariableOp,sequential_2/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_21/BiasAdd/ReadVariableOp-sequential_2/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_21/Conv2D/ReadVariableOp,sequential_2/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_26/BiasAdd/ReadVariableOp-sequential_2/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_25/Conv2D/ReadVariableOp,sequential_2/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_19/BiasAdd/ReadVariableOp-sequential_2/conv2d_19/BiasAdd/ReadVariableOp2^
-sequential_2/conv2d_24/BiasAdd/ReadVariableOp-sequential_2/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_22/Conv2D/ReadVariableOp,sequential_2/conv2d_22/Conv2D/ReadVariableOp: : : : : : : :/ +
)
_user_specified_nameconv2d_18_input: : : : : : : : :	 :
 : 
��
� 
!__inference__traced_restore_11279
file_prefix%
!assignvariableop_conv2d_18_kernel%
!assignvariableop_1_conv2d_18_bias'
#assignvariableop_2_conv2d_19_kernel%
!assignvariableop_3_conv2d_19_bias'
#assignvariableop_4_conv2d_20_kernel%
!assignvariableop_5_conv2d_20_bias'
#assignvariableop_6_conv2d_21_kernel%
!assignvariableop_7_conv2d_21_bias'
#assignvariableop_8_conv2d_22_kernel%
!assignvariableop_9_conv2d_22_bias(
$assignvariableop_10_conv2d_23_kernel&
"assignvariableop_11_conv2d_23_bias(
$assignvariableop_12_conv2d_24_kernel&
"assignvariableop_13_conv2d_24_bias(
$assignvariableop_14_conv2d_25_kernel&
"assignvariableop_15_conv2d_25_bias(
$assignvariableop_16_conv2d_26_kernel&
"assignvariableop_17_conv2d_26_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate
assignvariableop_23_total
assignvariableop_24_count/
+assignvariableop_25_adam_conv2d_18_kernel_m-
)assignvariableop_26_adam_conv2d_18_bias_m/
+assignvariableop_27_adam_conv2d_19_kernel_m-
)assignvariableop_28_adam_conv2d_19_bias_m/
+assignvariableop_29_adam_conv2d_20_kernel_m-
)assignvariableop_30_adam_conv2d_20_bias_m/
+assignvariableop_31_adam_conv2d_21_kernel_m-
)assignvariableop_32_adam_conv2d_21_bias_m/
+assignvariableop_33_adam_conv2d_22_kernel_m-
)assignvariableop_34_adam_conv2d_22_bias_m/
+assignvariableop_35_adam_conv2d_23_kernel_m-
)assignvariableop_36_adam_conv2d_23_bias_m/
+assignvariableop_37_adam_conv2d_24_kernel_m-
)assignvariableop_38_adam_conv2d_24_bias_m/
+assignvariableop_39_adam_conv2d_25_kernel_m-
)assignvariableop_40_adam_conv2d_25_bias_m/
+assignvariableop_41_adam_conv2d_26_kernel_m-
)assignvariableop_42_adam_conv2d_26_bias_m/
+assignvariableop_43_adam_conv2d_18_kernel_v-
)assignvariableop_44_adam_conv2d_18_bias_v/
+assignvariableop_45_adam_conv2d_19_kernel_v-
)assignvariableop_46_adam_conv2d_19_bias_v/
+assignvariableop_47_adam_conv2d_20_kernel_v-
)assignvariableop_48_adam_conv2d_20_bias_v/
+assignvariableop_49_adam_conv2d_21_kernel_v-
)assignvariableop_50_adam_conv2d_21_bias_v/
+assignvariableop_51_adam_conv2d_22_kernel_v-
)assignvariableop_52_adam_conv2d_22_bias_v/
+assignvariableop_53_adam_conv2d_23_kernel_v-
)assignvariableop_54_adam_conv2d_23_bias_v/
+assignvariableop_55_adam_conv2d_24_kernel_v-
)assignvariableop_56_adam_conv2d_24_bias_v/
+assignvariableop_57_adam_conv2d_25_kernel_v-
)assignvariableop_58_adam_conv2d_25_bias_v/
+assignvariableop_59_adam_conv2d_26_kernel_v-
)assignvariableop_60_adam_conv2d_26_bias_v
identity_62��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�"
RestoreV2/tensor_namesConst"/device:CPU:0*�!
value�!B�!=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:=�
RestoreV2/shape_and_slicesConst"/device:CPU:0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:=�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*K
dtypesA
?2=	*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:}
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_18_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_18_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_19_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_19_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_20_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_20_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_21_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_21_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_22_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_22_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_23_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_23_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_24_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_24_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_25_kernelIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_25_biasIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_26_kernelIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_26_biasIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0	*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0*
_output_shapes
 *
dtype0	P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:{
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0{
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_18_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_18_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_19_kernel_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_19_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_20_kernel_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
_output_shapes
:*
T0�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_20_bias_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_21_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_21_bias_mIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_22_kernel_mIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_22_bias_mIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_23_kernel_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_23_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype0P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_24_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype0P
Identity_38IdentityRestoreV2:tensors:38*
_output_shapes
:*
T0�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_24_bias_mIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_25_kernel_mIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
_output_shapes
:*
T0�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_25_bias_mIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_26_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype0P
Identity_42IdentityRestoreV2:tensors:42*
_output_shapes
:*
T0�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_26_bias_mIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_18_kernel_vIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_18_bias_vIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
_output_shapes
:*
T0�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_19_kernel_vIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_19_bias_vIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_20_kernel_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_20_bias_vIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_21_kernel_vIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
_output_shapes
:*
T0�
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_21_bias_vIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_22_kernel_vIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T0�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_22_bias_vIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_23_kernel_vIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_23_bias_vIdentity_54:output:0*
dtype0*
_output_shapes
 P
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_24_kernel_vIdentity_55:output:0*
dtype0*
_output_shapes
 P
Identity_56IdentityRestoreV2:tensors:56*
_output_shapes
:*
T0�
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_24_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype0P
Identity_57IdentityRestoreV2:tensors:57*
_output_shapes
:*
T0�
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_25_kernel_vIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_25_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype0P
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_26_kernel_vIdentity_59:output:0*
dtype0*
_output_shapes
 P
Identity_60IdentityRestoreV2:tensors:60*
_output_shapes
:*
T0�
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_26_bias_vIdentity_60:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_61Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0�
Identity_62IdentityIdentity_61:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_62Identity_62:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_59: :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : 
��
�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9892
conv2d_18_input,
(conv2d_18_statefulpartitionedcall_args_1,
(conv2d_18_statefulpartitionedcall_args_2,
(conv2d_19_statefulpartitionedcall_args_1,
(conv2d_19_statefulpartitionedcall_args_2,
(conv2d_20_statefulpartitionedcall_args_1,
(conv2d_20_statefulpartitionedcall_args_2,
(conv2d_21_statefulpartitionedcall_args_1,
(conv2d_21_statefulpartitionedcall_args_2,
(conv2d_22_statefulpartitionedcall_args_1,
(conv2d_22_statefulpartitionedcall_args_2,
(conv2d_23_statefulpartitionedcall_args_1,
(conv2d_23_statefulpartitionedcall_args_2,
(conv2d_24_statefulpartitionedcall_args_1,
(conv2d_24_statefulpartitionedcall_args_2,
(conv2d_25_statefulpartitionedcall_args_1,
(conv2d_25_statefulpartitionedcall_args_2,
(conv2d_26_statefulpartitionedcall_args_1,
(conv2d_26_statefulpartitionedcall_args_2
identity��!conv2d_18/StatefulPartitionedCall�2conv2d_18/kernel/Regularizer/Square/ReadVariableOp�!conv2d_19/StatefulPartitionedCall�2conv2d_19/kernel/Regularizer/Square/ReadVariableOp�!conv2d_20/StatefulPartitionedCall�2conv2d_20/kernel/Regularizer/Square/ReadVariableOp�!conv2d_21/StatefulPartitionedCall�2conv2d_21/kernel/Regularizer/Square/ReadVariableOp�!conv2d_22/StatefulPartitionedCall�2conv2d_22/kernel/Regularizer/Square/ReadVariableOp�!conv2d_23/StatefulPartitionedCall�2conv2d_23/kernel/Regularizer/Square/ReadVariableOp�!conv2d_24/StatefulPartitionedCall�2conv2d_24/kernel/Regularizer/Square/ReadVariableOp�!conv2d_25/StatefulPartitionedCall�2conv2d_25/kernel/Regularizer/Square/ReadVariableOp�!conv2d_26/StatefulPartitionedCall�2conv2d_26/kernel/Regularizer/Square/ReadVariableOp�
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallconv2d_18_input(conv2d_18_statefulpartitionedcall_args_1(conv2d_18_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9507*L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_9501*
Tout
2�
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0(conv2d_19_statefulpartitionedcall_args_1(conv2d_19_statefulpartitionedcall_args_2*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9539*L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_9533*
Tout
2**
config_proto

CPU

GPU 2J 8�
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0(conv2d_20_statefulpartitionedcall_args_1(conv2d_20_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9571*L
fGRE
C__inference_conv2d_20_layer_call_and_return_conditional_losses_9565*
Tout
2�
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0(conv2d_21_statefulpartitionedcall_args_1(conv2d_21_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9603*L
fGRE
C__inference_conv2d_21_layer_call_and_return_conditional_losses_9597�
max_pooling2d_2/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9622*R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9616*
Tout
2�
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0(conv2d_22_statefulpartitionedcall_args_1(conv2d_22_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_22_layer_call_and_return_conditional_losses_9646*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:����������� *
Tin
2*+
_gradient_op_typePartitionedCall-9652�
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0(conv2d_23_statefulpartitionedcall_args_1(conv2d_23_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9684*L
fGRE
C__inference_conv2d_23_layer_call_and_return_conditional_losses_9678*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:����������� *
Tin
2�
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0(conv2d_24_statefulpartitionedcall_args_1(conv2d_24_statefulpartitionedcall_args_2*
Tin
2*1
_output_shapes
:����������� *+
_gradient_op_typePartitionedCall-9716*L
fGRE
C__inference_conv2d_24_layer_call_and_return_conditional_losses_9710*
Tout
2**
config_proto

CPU

GPU 2J 8�
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0(conv2d_25_statefulpartitionedcall_args_1(conv2d_25_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-9748*L
fGRE
C__inference_conv2d_25_layer_call_and_return_conditional_losses_9742*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:����������� �
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0(conv2d_26_statefulpartitionedcall_args_1(conv2d_26_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*1
_output_shapes
:�����������*+
_gradient_op_typePartitionedCall-9780*L
fGRE
C__inference_conv2d_26_layer_call_and_return_conditional_losses_9774*
Tout
2�
2conv2d_18/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_18_statefulpartitionedcall_args_1"^conv2d_18/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_18/kernel/Regularizer/SquareSquare:conv2d_18/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_18/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_18/kernel/Regularizer/SumSum'conv2d_18/kernel/Regularizer/Square:y:0+conv2d_18/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_18/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/mulMul+conv2d_18/kernel/Regularizer/mul/x:output:0)conv2d_18/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_18/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_18/kernel/Regularizer/addAddV2+conv2d_18/kernel/Regularizer/add/x:output:0$conv2d_18/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_19/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_19_statefulpartitionedcall_args_1"^conv2d_19/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_19/kernel/Regularizer/SquareSquare:conv2d_19/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_19/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_19/kernel/Regularizer/SumSum'conv2d_19/kernel/Regularizer/Square:y:0+conv2d_19/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/mulMul+conv2d_19/kernel/Regularizer/mul/x:output:0)conv2d_19/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_19/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_19/kernel/Regularizer/addAddV2+conv2d_19/kernel/Regularizer/add/x:output:0$conv2d_19/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_20_statefulpartitionedcall_args_1"^conv2d_20/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:*
dtype0�
#conv2d_20/kernel/Regularizer/SquareSquare:conv2d_20/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:*
T0{
"conv2d_20/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_20/kernel/Regularizer/SumSum'conv2d_20/kernel/Regularizer/Square:y:0+conv2d_20/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_20/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_20/kernel/Regularizer/mulMul+conv2d_20/kernel/Regularizer/mul/x:output:0)conv2d_20/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_20/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0�
 conv2d_20/kernel/Regularizer/addAddV2+conv2d_20/kernel/Regularizer/add/x:output:0$conv2d_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_21_statefulpartitionedcall_args_1"^conv2d_21/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:�
#conv2d_21/kernel/Regularizer/SquareSquare:conv2d_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:{
"conv2d_21/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_21/kernel/Regularizer/SumSum'conv2d_21/kernel/Regularizer/Square:y:0+conv2d_21/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0�
 conv2d_21/kernel/Regularizer/mulMul+conv2d_21/kernel/Regularizer/mul/x:output:0)conv2d_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_21/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_21/kernel/Regularizer/addAddV2+conv2d_21/kernel/Regularizer/add/x:output:0$conv2d_21/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_22_statefulpartitionedcall_args_1"^conv2d_22/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_22/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_22/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_22/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_23_statefulpartitionedcall_args_1"^conv2d_23/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_23/kernel/Regularizer/SquareSquare:conv2d_23/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_23/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_23/kernel/Regularizer/SumSum'conv2d_23/kernel/Regularizer/Square:y:0+conv2d_23/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_23/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<�
 conv2d_23/kernel/Regularizer/mulMul+conv2d_23/kernel/Regularizer/mul/x:output:0)conv2d_23/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_23/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_23/kernel/Regularizer/addAddV2+conv2d_23/kernel/Regularizer/add/x:output:0$conv2d_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
2conv2d_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_24_statefulpartitionedcall_args_1"^conv2d_24/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_24/kernel/Regularizer/SquareSquare:conv2d_24/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_24/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_24/kernel/Regularizer/SumSum'conv2d_24/kernel/Regularizer/Square:y:0+conv2d_24/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_24/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0)conv2d_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_24/kernel/Regularizer/addAddV2+conv2d_24/kernel/Regularizer/add/x:output:0$conv2d_24/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_25_statefulpartitionedcall_args_1"^conv2d_25/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  {
"conv2d_25/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_25/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_25/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_26_statefulpartitionedcall_args_1"^conv2d_26/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_26/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_26/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    �
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall3^conv2d_18/kernel/Regularizer/Square/ReadVariableOp"^conv2d_19/StatefulPartitionedCall3^conv2d_19/kernel/Regularizer/Square/ReadVariableOp"^conv2d_20/StatefulPartitionedCall3^conv2d_20/kernel/Regularizer/Square/ReadVariableOp"^conv2d_21/StatefulPartitionedCall3^conv2d_21/kernel/Regularizer/Square/ReadVariableOp"^conv2d_22/StatefulPartitionedCall3^conv2d_22/kernel/Regularizer/Square/ReadVariableOp"^conv2d_23/StatefulPartitionedCall3^conv2d_23/kernel/Regularizer/Square/ReadVariableOp"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/Square/ReadVariableOp"^conv2d_25/StatefulPartitionedCall3^conv2d_25/kernel/Regularizer/Square/ReadVariableOp"^conv2d_26/StatefulPartitionedCall3^conv2d_26/kernel/Regularizer/Square/ReadVariableOp*1
_output_shapes
:�����������*
T0"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2h
2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2conv2d_21/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2h
2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2h
2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2conv2d_23/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2conv2d_18/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2conv2d_24/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2conv2d_19/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2h
2conv2d_20/kernel/Regularizer/Square/ReadVariableOp2conv2d_20/kernel/Regularizer/Square/ReadVariableOp:/ +
)
_user_specified_nameconv2d_18_input: : : : : : : : :	 :
 : : : : : : : : 
�
�
__inference_loss_fn_7_10858?
;conv2d_25_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_25/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_25_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:  �
#conv2d_25/kernel/Regularizer/SquareSquare:conv2d_25/kernel/Regularizer/Square/ReadVariableOp:value:0*&
_output_shapes
:  *
T0{
"conv2d_25/kernel/Regularizer/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:�
 conv2d_25/kernel/Regularizer/SumSum'conv2d_25/kernel/Regularizer/Square:y:0+conv2d_25/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_25/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/mulMul+conv2d_25/kernel/Regularizer/mul/x:output:0)conv2d_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_25/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_25/kernel/Regularizer/addAddV2+conv2d_25/kernel/Regularizer/add/x:output:0$conv2d_25/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0�
IdentityIdentity$conv2d_25/kernel/Regularizer/add:z:03^conv2d_25/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_25/kernel/Regularizer/Square/ReadVariableOp2conv2d_25/kernel/Regularizer/Square/ReadVariableOp:  
�
�
,__inference_sequential_2_layer_call_fn_10645

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18*
Tout
2**
config_proto

CPU

GPU 2J 8*1
_output_shapes
:�����������*
Tin
2*,
_gradient_op_typePartitionedCall-10106*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_10105�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : 
�
�
C__inference_conv2d_22_layer_call_and_return_conditional_losses_9646

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�2conv2d_22/kernel/Regularizer/Square/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingSAME*A
_output_shapes/
-:+��������������������������� *
T0*
strides
�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� �
2conv2d_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_22/kernel/Regularizer/SquareSquare:conv2d_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_22/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             �
 conv2d_22/kernel/Regularizer/SumSum'conv2d_22/kernel/Regularizer/Square:y:0+conv2d_22/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_22/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/mulMul+conv2d_22/kernel/Regularizer/mul/x:output:0)conv2d_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: g
"conv2d_22/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_22/kernel/Regularizer/addAddV2+conv2d_22/kernel/Regularizer/add/x:output:0$conv2d_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_22/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_22/kernel/Regularizer/Square/ReadVariableOp2conv2d_22/kernel/Regularizer/Square/ReadVariableOp: : :& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_8_10873?
;conv2d_26_kernel_regularizer_square_readvariableop_resource
identity��2conv2d_26/kernel/Regularizer/Square/ReadVariableOp�
2conv2d_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_26_kernel_regularizer_square_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
#conv2d_26/kernel/Regularizer/SquareSquare:conv2d_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: {
"conv2d_26/kernel/Regularizer/ConstConst*
_output_shapes
:*%
valueB"             *
dtype0�
 conv2d_26/kernel/Regularizer/SumSum'conv2d_26/kernel/Regularizer/Square:y:0+conv2d_26/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/mul/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/mulMul+conv2d_26/kernel/Regularizer/mul/x:output:0)conv2d_26/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0g
"conv2d_26/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
 conv2d_26/kernel/Regularizer/addAddV2+conv2d_26/kernel/Regularizer/add/x:output:0$conv2d_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
IdentityIdentity$conv2d_26/kernel/Regularizer/add:z:03^conv2d_26/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_26/kernel/Regularizer/Square/ReadVariableOp2conv2d_26/kernel/Regularizer/Square/ReadVariableOp:  "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
U
conv2d_18_inputB
!serving_default_conv2d_18_input:0�����������G
	conv2d_26:
StatefulPartitionedCall:0�����������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:�
�f
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer_with_weights-8
layer-10
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�a
_tf_keras_sequential�a{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "batch_input_shape": [null, 512, 512, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "batch_input_shape": [null, 512, 512, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["acc"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "conv2d_18_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 512, 512, 3], "config": {"batch_input_shape": [null, 512, 512, 3], "dtype": "float32", "sparse": false, "name": "conv2d_18_input"}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 512, 512, 3], "config": {"name": "conv2d_18", "trainable": true, "batch_input_shape": [null, 512, 512, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
�

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
�

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratem�m�m�m�"m�#m�(m�)m�2m�3m�8m�9m�>m�?m�Dm�Em�Jm�Km�v�v�v�v�"v�#v�(v�)v�2v�3v�8v�9v�>v�?v�Dv�Ev�Jv�Kv�"
	optimizer
�
0
1
2
3
"4
#5
(6
)7
28
39
810
911
>12
?13
D14
E15
J16
K17"
trackable_list_wrapper
h
�0
�1
�2
�3
�4
�5
�6
�7
�8"
trackable_list_wrapper
�
0
1
2
3
"4
#5
(6
)7
28
39
810
911
>12
?13
D14
E15
J16
K17"
trackable_list_wrapper
�
Umetrics
	variables
regularization_losses
trainable_variables

Vlayers
Wlayer_regularization_losses
Xnon_trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ymetrics
	variables

Zlayers
trainable_variables
regularization_losses
[layer_regularization_losses
\non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_18/kernel
:2conv2d_18/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
]metrics
	variables

^layers
trainable_variables
regularization_losses
_layer_regularization_losses
`non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_19/kernel
:2conv2d_19/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
ametrics
	variables

blayers
trainable_variables
 regularization_losses
clayer_regularization_losses
dnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_20/kernel
:2conv2d_20/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
emetrics
$	variables

flayers
%trainable_variables
&regularization_losses
glayer_regularization_losses
hnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_21/kernel
:2conv2d_21/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
imetrics
*	variables

jlayers
+trainable_variables
,regularization_losses
klayer_regularization_losses
lnon_trainable_variables
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
mmetrics
.	variables

nlayers
/trainable_variables
0regularization_losses
olayer_regularization_losses
pnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_22/kernel
: 2conv2d_22/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
qmetrics
4	variables

rlayers
5trainable_variables
6regularization_losses
slayer_regularization_losses
tnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_23/kernel
: 2conv2d_23/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
umetrics
:	variables

vlayers
;trainable_variables
<regularization_losses
wlayer_regularization_losses
xnon_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_24/kernel
: 2conv2d_24/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
ymetrics
@	variables

zlayers
Atrainable_variables
Bregularization_losses
{layer_regularization_losses
|non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_25/kernel
: 2conv2d_25/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
}metrics
F	variables

~layers
Gtrainable_variables
Hregularization_losses
layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_26/kernel
:2conv2d_26/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�metrics
L	variables
�layers
Mtrainable_variables
Nregularization_losses
 �layer_regularization_losses
�non_trainable_variables
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
(
�0"
trackable_list_wrapper
f
0
1
2
3
4
5
6
	7

8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
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
trackable_list_wrapper
�

�total

�count
�
_fn_kwargs
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "acc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "acc", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�	variables
�layers
�trainable_variables
�regularization_losses
 �layer_regularization_losses
�non_trainable_variables
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
0
�0
�1"
trackable_list_wrapper
/:-2Adam/conv2d_18/kernel/m
!:2Adam/conv2d_18/bias/m
/:-2Adam/conv2d_19/kernel/m
!:2Adam/conv2d_19/bias/m
/:-2Adam/conv2d_20/kernel/m
!:2Adam/conv2d_20/bias/m
/:-2Adam/conv2d_21/kernel/m
!:2Adam/conv2d_21/bias/m
/:- 2Adam/conv2d_22/kernel/m
!: 2Adam/conv2d_22/bias/m
/:-  2Adam/conv2d_23/kernel/m
!: 2Adam/conv2d_23/bias/m
/:-  2Adam/conv2d_24/kernel/m
!: 2Adam/conv2d_24/bias/m
/:-  2Adam/conv2d_25/kernel/m
!: 2Adam/conv2d_25/bias/m
/:- 2Adam/conv2d_26/kernel/m
!:2Adam/conv2d_26/bias/m
/:-2Adam/conv2d_18/kernel/v
!:2Adam/conv2d_18/bias/v
/:-2Adam/conv2d_19/kernel/v
!:2Adam/conv2d_19/bias/v
/:-2Adam/conv2d_20/kernel/v
!:2Adam/conv2d_20/bias/v
/:-2Adam/conv2d_21/kernel/v
!:2Adam/conv2d_21/bias/v
/:- 2Adam/conv2d_22/kernel/v
!: 2Adam/conv2d_22/bias/v
/:-  2Adam/conv2d_23/kernel/v
!: 2Adam/conv2d_23/bias/v
/:-  2Adam/conv2d_24/kernel/v
!: 2Adam/conv2d_24/bias/v
/:-  2Adam/conv2d_25/kernel/v
!: 2Adam/conv2d_25/bias/v
/:- 2Adam/conv2d_26/kernel/v
!:2Adam/conv2d_26/bias/v
�2�
,__inference_sequential_2_layer_call_fn_10127
,__inference_sequential_2_layer_call_fn_10668
,__inference_sequential_2_layer_call_fn_10257
,__inference_sequential_2_layer_call_fn_10645�
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
�2�
F__inference_sequential_2_layer_call_and_return_conditional_losses_9892
G__inference_sequential_2_layer_call_and_return_conditional_losses_10491
G__inference_sequential_2_layer_call_and_return_conditional_losses_10622
F__inference_sequential_2_layer_call_and_return_conditional_losses_9998�
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
__inference__wrapped_model_9480�
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
annotations� *8�5
3�0
conv2d_18_input�����������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
(__inference_conv2d_18_layer_call_fn_9512�
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
annotations� *7�4
2�/+���������������������������
�2�
C__inference_conv2d_18_layer_call_and_return_conditional_losses_9501�
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
annotations� *7�4
2�/+���������������������������
�2�
(__inference_conv2d_19_layer_call_fn_9544�
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
annotations� *7�4
2�/+���������������������������
�2�
C__inference_conv2d_19_layer_call_and_return_conditional_losses_9533�
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
annotations� *7�4
2�/+���������������������������
�2�
(__inference_conv2d_20_layer_call_fn_9576�
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
annotations� *7�4
2�/+���������������������������
�2�
C__inference_conv2d_20_layer_call_and_return_conditional_losses_9565�
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
annotations� *7�4
2�/+���������������������������
�2�
(__inference_conv2d_21_layer_call_fn_9608�
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
annotations� *7�4
2�/+���������������������������
�2�
C__inference_conv2d_21_layer_call_and_return_conditional_losses_9597�
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
annotations� *7�4
2�/+���������������������������
�2�
.__inference_max_pooling2d_2_layer_call_fn_9625�
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
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9616�
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
(__inference_conv2d_22_layer_call_fn_9657�
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
annotations� *7�4
2�/+���������������������������
�2�
C__inference_conv2d_22_layer_call_and_return_conditional_losses_9646�
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
annotations� *7�4
2�/+���������������������������
�2�
(__inference_conv2d_23_layer_call_fn_9689�
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
annotations� *7�4
2�/+��������������������������� 
�2�
C__inference_conv2d_23_layer_call_and_return_conditional_losses_9678�
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
annotations� *7�4
2�/+��������������������������� 
�2�
(__inference_conv2d_24_layer_call_fn_9721�
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
annotations� *7�4
2�/+��������������������������� 
�2�
C__inference_conv2d_24_layer_call_and_return_conditional_losses_9710�
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
annotations� *7�4
2�/+��������������������������� 
�2�
(__inference_conv2d_25_layer_call_fn_9753�
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
annotations� *7�4
2�/+��������������������������� 
�2�
C__inference_conv2d_25_layer_call_and_return_conditional_losses_9742�
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
annotations� *7�4
2�/+��������������������������� 
�2�
(__inference_conv2d_26_layer_call_fn_9785�
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
annotations� *7�4
2�/+��������������������������� 
�2�
C__inference_conv2d_26_layer_call_and_return_conditional_losses_9774�
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
annotations� *7�4
2�/+��������������������������� 
�2�
__inference_loss_fn_0_10753�
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
__inference_loss_fn_1_10768�
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
__inference_loss_fn_2_10783�
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
__inference_loss_fn_3_10798�
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
__inference_loss_fn_4_10813�
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
__inference_loss_fn_5_10828�
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
__inference_loss_fn_6_10843�
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
__inference_loss_fn_7_10858�
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
__inference_loss_fn_8_10873�
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
:B8
#__inference_signature_wrapper_10358conv2d_18_input
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
(__inference_conv2d_23_layer_call_fn_9689�89I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
G__inference_sequential_2_layer_call_and_return_conditional_losses_10622�"#()2389>?DEJKA�>
7�4
*�'
inputs�����������
p 

 
� "/�,
%�"
0�����������
� :
__inference_loss_fn_0_10753�

� 
� "� �
C__inference_conv2d_18_layer_call_and_return_conditional_losses_9501�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
C__inference_conv2d_23_layer_call_and_return_conditional_losses_9678�89I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
(__inference_conv2d_19_layer_call_fn_9544�I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������:
__inference_loss_fn_6_10843>�

� 
� "� �
(__inference_conv2d_26_layer_call_fn_9785�JKI�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
C__inference_conv2d_22_layer_call_and_return_conditional_losses_9646�23I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9616�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
F__inference_sequential_2_layer_call_and_return_conditional_losses_9998�"#()2389>?DEJKJ�G
@�=
3�0
conv2d_18_input�����������
p 

 
� "/�,
%�"
0�����������
� �
#__inference_signature_wrapper_10358�"#()2389>?DEJKU�R
� 
K�H
F
conv2d_18_input3�0
conv2d_18_input�����������"?�<
:
	conv2d_26-�*
	conv2d_26�����������:
__inference_loss_fn_1_10768�

� 
� "� �
,__inference_sequential_2_layer_call_fn_10127�"#()2389>?DEJKJ�G
@�=
3�0
conv2d_18_input�����������
p

 
� ""�������������
C__inference_conv2d_26_layer_call_and_return_conditional_losses_9774�JKI�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
__inference__wrapped_model_9480�"#()2389>?DEJKB�?
8�5
3�0
conv2d_18_input�����������
� "?�<
:
	conv2d_26-�*
	conv2d_26������������
,__inference_sequential_2_layer_call_fn_10645{"#()2389>?DEJKA�>
7�4
*�'
inputs�����������
p

 
� ""�������������
(__inference_conv2d_21_layer_call_fn_9608�()I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
C__inference_conv2d_25_layer_call_and_return_conditional_losses_9742�DEI�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� :
__inference_loss_fn_7_10858D�

� 
� "� :
__inference_loss_fn_2_10783"�

� 
� "� �
C__inference_conv2d_24_layer_call_and_return_conditional_losses_9710�>?I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
C__inference_conv2d_21_layer_call_and_return_conditional_losses_9597�()I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
.__inference_max_pooling2d_2_layer_call_fn_9625�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
(__inference_conv2d_20_layer_call_fn_9576�"#I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
,__inference_sequential_2_layer_call_fn_10668{"#()2389>?DEJKA�>
7�4
*�'
inputs�����������
p 

 
� ""������������:
__inference_loss_fn_4_108132�

� 
� "� �
C__inference_conv2d_20_layer_call_and_return_conditional_losses_9565�"#I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� :
__inference_loss_fn_8_10873J�

� 
� "� �
(__inference_conv2d_24_layer_call_fn_9721�>?I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
G__inference_sequential_2_layer_call_and_return_conditional_losses_10491�"#()2389>?DEJKA�>
7�4
*�'
inputs�����������
p

 
� "/�,
%�"
0�����������
� �
F__inference_sequential_2_layer_call_and_return_conditional_losses_9892�"#()2389>?DEJKJ�G
@�=
3�0
conv2d_18_input�����������
p

 
� "/�,
%�"
0�����������
� �
(__inference_conv2d_22_layer_call_fn_9657�23I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� :
__inference_loss_fn_3_10798(�

� 
� "� �
(__inference_conv2d_18_layer_call_fn_9512�I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������:
__inference_loss_fn_5_108288�

� 
� "� �
C__inference_conv2d_19_layer_call_and_return_conditional_losses_9533�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
(__inference_conv2d_25_layer_call_fn_9753�DEI�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
,__inference_sequential_2_layer_call_fn_10257�"#()2389>?DEJKJ�G
@�=
3�0
conv2d_18_input�����������
p 

 
� ""������������