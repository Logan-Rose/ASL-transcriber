$	f0????'@?Y?"E9@V?Z? @!??@???a@	!       "^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??p??0	@V(??????A?U????@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?
?b??@?|y?ѵ?A?>Ȳ`@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??1 {?*@?? @????A???^?*@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???U??@ZJ??P???AbJ$??@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???v?A@?N?j???A8H??A@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?yƾd1@??T[??AB%?c\?0@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsQ0c
?@˃?9D??AB^&??@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???K?:@???x???A?k?6@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	+N??&@Q/?4'O@Ad> ЙT@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
??,z??"@C8fٓ?@A?B?5vi@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?{?_?P@?Y??????A??\??@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsV?Z? @'?
b???A???	 @"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsByG@k?MG?@A????E@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?q??>?@?TPQ?+??AZGUD? @"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?#~?^@=Y???@A?7???@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails`YiR
:
@??3??@??A>Z?1??@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails.Ȗ?k@?t???A?*n?b@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails? ?m?8@C??up???A'1??@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?WXp??@?A?f???A?8d??@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??@???a@???f?a@A??9#J[@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB??	ܚ@?4?($???A?n??I?@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsk????1@`???*@A?)H?@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsQ?f??%@??N@?@A?K???@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???sr@l%t??Y??A?????@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?????@????????A????s?@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??f?vG@?;????@A;??@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?4~??+@?TO?}??A?????*@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?O???@??%?L1??A.???@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7??5|'@B_z?s???A,d???&@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?B?Y??@???3??A%;6?@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?)t^c? @#k??"??AץF?g???*	?MbPPW A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?r?SR??@!?3j?X@)?r?SR??@1?3j?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?@.q䁠?!Vc? ??x?)?@.q䁠?1Vc? ??x?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism%?j????!6&҂8??)~??7L??1T?3?OS^?:Preprocessing2F
Iterator::ModelV??Dׅ??!??@?h???)Ü?Mo?1:???2G?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?唀X??@!???ls?X@)Է???h?1???]PsB?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI      Y@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	L???Hx@l?*/9@?N?j???!???f?a@	!       "	!       *	!       2$	?}9˴@?8?c;@ץF?g???!8H??A@:	!       B	!       J	!       R	!       Z	!       b	!       JCPU_ONLYb q      Y@