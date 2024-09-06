# 2. Fairness with slice finder: does not exist in mlwhatif. first, we need a new patch type on the outputs of the
# model test set predictions joined with the input data. this join kind of requires provenance alreday.
# does it???? we can just rely on the order not changing between featurisation and before. however, attributes might
# be removed during this process. or do we assume they don't?
#  once we have a problematic slice, we need to go back and apply it as a filter again, which again requires
#  provenance. maybe we need to add basic provenance support first? or not!! this filter can be computed using
#  the unfeaturised state of the data by relying on the order not changing. then we can just compute a mask and
#  apply it to the unfeaturised data.