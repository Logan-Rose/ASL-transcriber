	?F???@?F???@!?F???@	?#iy?Ӻ??#iy?Ӻ?!?#iy?Ӻ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?F???@'???C??A?c?}?@Y?f??6???*	l???Z?A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatora?xw?f@!??{[??X@)a?xw?f@1??{[??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5??o?h??!?LӴ? ??)?	i?A'??1? VƇk??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchbHN&n??!?x?ϝ???)bHN&n??1?x?ϝ???:Preprocessing2F
Iterator::Model????ˍ??!???px5??)l??g??r?1?N???d?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapC???f@!?`??X@)??ek}a?1f????S?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?#iy?Ӻ?I??!K?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	'???C??'???C??!'???C??      ??!       "      ??!       *      ??!       2	?c?}?@?c?}?@!?c?}?@:      ??!       B      ??!       J	?f??6????f??6???!?f??6???R      ??!       Z	?f??6????f??6???!?f??6???b      ??!       JCPU_ONLYY?#iy?Ӻ?b q??!K?X@