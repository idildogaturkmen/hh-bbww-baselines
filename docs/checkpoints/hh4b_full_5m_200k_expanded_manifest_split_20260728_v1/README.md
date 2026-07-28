# HH4b full 5M-background/200k-signal expanded manifest

This checkpoint freezes the complete 630-member source-to-candidate
registry after closure of all 27 ttbar canonical-schema holds.

The source population remains exactly 5,000,000 generated background
events and 200,000 generated signal events. The old 581-member
development population excluded the 27 ttbar holds. Their closure
raises the canonical non-test population to 608 members.

The final expanded-v2 grouped split contains 464 train members, 121
validation members, and 45 final-evaluation members. The development
population therefore contains 585 members. Every source member is one
indivisible group and no member crosses splits.

The final evaluation manifest preserves the 22 original sealed members
and adopts the previously proposed 23-member deterministic
metadata-only coverage supplement. The 22 original members remain
historically pristine. The supplement is explicitly recorded as not
historically pristine because it belonged to the earlier development
population. All 45 members are closed to the new expanded-v2 models
from this checkpoint onward.

This freeze opened no candidate file and read no candidate row. It
trained no model, calculated no score, selected no threshold, and
performed no physical normalization.

The next gate is to commit this manifest and split policy, then
materialize and validate one common development candidate/feature
cache for every baseline model.
