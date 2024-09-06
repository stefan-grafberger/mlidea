# 3. robustness:
# we can just run a corruption udf on everything that changes a small part of the data, then we run a diff
# detection and use it to create a mask. only on the changed ones do we need to try a correction. this needs to
# be modeled in the DAG, since we don't want to run the correction on all data in this first step.
# however, not perfectly accurate results
#
# also, the provenance part should be possible to turn off for performance comparisons. in general, we do need
# provenance, but with enough simplyfying assumptions about the order not changing and all data being available
# until right before the featurisation, we can get away without. maybe I wasted a day today... or we still build
# it to have better explanations?