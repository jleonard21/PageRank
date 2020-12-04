# Pagerank HW: Web Search with Word2Vec

In this assignment, we create a search engine that uses both pagerank and word embeddings to find high quality results.

Task 1 is required, Tasks 2 and 3 are extra credit.

**Due date:** Sunday, 6 December at midnight

## Task 1:

The file `pagerank.py` contains a completed solution to the pagerank problem. Recall that we can use this file to find the "highest quality" search results from the https://lawfareblog.com website.

For example, searching for "weapons" gives us:

**Part 1:**
To check that your implementation is working,
you should run the program on the `small.csv.gz` graph which is the example graph from the *Deeper Inside Pagerank* paper.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=./lawfareblog.csv.gz --search_query='weapons'
INFO:root:rank=0 pagerank=0.0011619674041867256 url=www.lawfareblog.com/slaughterbots-and-other-anticipated-autonomous-weapons-problems
INFO:root:rank=1 pagerank=0.0006675782497040927 url=www.lawfareblog.com/history-do-it-yourself-weapons-and-explosives-manuals-america
INFO:root:rank=2 pagerank=0.0006599930347874761 url=www.lawfareblog.com/lethal-autonomous-weapons-systems-first-and-second-un-gge-meetings
INFO:root:rank=3 pagerank=0.0006531275575980544 url=www.lawfareblog.com/lethal-autonomous-weapons-systems-recent-developments
INFO:root:rank=4 pagerank=0.0006407579057849944 url=www.lawfareblog.com/too-early-ban-us-and-uk-positions-lethal-autonomous-weapons-systems
INFO:root:rank=5 pagerank=0.0006338492385111749 url=www.lawfareblog.com/living-weapons-biological-warfare-and-international-security-gregory-koblentz
INFO:root:rank=6 pagerank=0.0005889176391065121 url=www.lawfareblog.com/critical-gaps-remain-defense-department-weapons-system-cybersecurity
INFO:root:rank=7 pagerank=0.0005780397332273424 url=www.lawfareblog.com/digital-strangelove-cyber-dangers-nuclear-weapons
INFO:root:rank=8 pagerank=0.0005346386460587382 url=www.lawfareblog.com/chemical-weapons-syria-enough-justify-use-force
INFO:root:rank=9 pagerank=0.0005290458793751895 url=www.lawfareblog.com/complexities-usg-covert-action-supply-weapons-syrian-rebels
```
and searching for "drones" gives us:
```
$ python3 pagerank.py --data=./lawfareblog.csv.gz --search_query='drones'
INFO:root:rank=0 pagerank=0.0006651429575867951 url=www.lawfareblog.com/future-violence-robots-and-germs-hackers-and-drones-confronting-new-age-threat
INFO:root:rank=1 pagerank=0.000632093520835042 url=www.lawfareblog.com/faa-wants-hear-you-about-privacy-and-domestic-drones
INFO:root:rank=2 pagerank=0.0006249933503568172 url=www.lawfareblog.com/ryan-calo-faas-setback-domestic-drones
INFO:root:rank=3 pagerank=0.0005648414953611791 url=www.lawfareblog.com/future-violence-now-hostile-use-drones
INFO:root:rank=4 pagerank=0.0005380361108109355 url=www.lawfareblog.com/lawfare-podcast-episode-5-missy-cummings-drones-drones-drones
INFO:root:rank=5 pagerank=0.0005302542704157531 url=www.lawfareblog.com/transcript-john-brennans-speech-yemen-and-drones
INFO:root:rank=6 pagerank=0.0005287894746288657 url=www.lawfareblog.com/no-more-drones-cia
INFO:root:rank=7 pagerank=0.0005287464591674507 url=www.lawfareblog.com/defending-drones-oxford-union
INFO:root:rank=8 pagerank=0.0005245042848400772 url=www.lawfareblog.com/readings-civilian-intelligence-agencies-and-use-armed-drones-ian-henderson
INFO:root:rank=9 pagerank=0.0005204146727919579 url=www.lawfareblog.com/wittes-v-oconnell-targeted-killing-and-drones
```
and searching for "targets" gives us:

```
$ python3 pagerank.py --data=./lawfareblog.csv.gz --search_query='targets'
INFO:root:rank=0 pagerank=0.0005385450785979629 url=www.lawfareblog.com/exclusive-nsa-program-can-target-thoughts-millions-targets-thousands-americans
INFO:root:rank=1 pagerank=0.0005327311810106039 url=www.lawfareblog.com/saudi-women-granted-new-rights-kurdistan-under-pressure-after-referendum-egypt-targets-dissidents
INFO:root:rank=2 pagerank=0.0005264155333861709 url=www.lawfareblog.com/turkeys-referendum-pushes-country-toward-authoritarianism-attack-targets-civilians-syrian-ceasefire
INFO:root:rank=3 pagerank=0.0005200763698667288 url=www.lawfareblog.com/british-anwar-al-awlaki-scenario-uk-targets-british-isil-member-syria-imminentcontinuous-threat
INFO:root:rank=4 pagerank=0.0005126169417053461 url=www.lawfareblog.com/middle-east-ticker-mosul-offensive-begins-us-fires-targets-yemen-and-what-intervention-syria-will
INFO:root:rank=5 pagerank=0.0005046890000812709 url=www.lawfareblog.com/us-forces-said-have-bombed-isis-targets-iraq
INFO:root:rank=6 pagerank=0.0005030750180594623 url=www.lawfareblog.com/trumps-big-middle-east-trip-us-targets-assad-forces-syria-and-iran-goes-polls
INFO:root:rank=7 pagerank=0.0005025876453146338 url=www.lawfareblog.com/international-fallout-trumps-immigration-and-refugee-order-us-raid-targets-aqap-yemen-greece-turkey
INFO:root:rank=8 pagerank=0.0005012631299905479 url=www.lawfareblog.com/der-spiegel-claims-germany-witholds-intel-militants-who-might-be-drone-strike-targets
INFO:root:rank=9 pagerank=0.0005012631299905479 url=www.lawfareblog.com/what-right-number-call-detail-records-42-targets-under-fisas-business-records-authority
```
Notice that the results for each of these terms do not overlap at all.

In a real search engine, however, we probably want to return results that mention "drones" or "targets" if someone searches for "weapons" since these terms are highly related. Word vectors give us a tool for doing this.

In particular, we can use the gensim library and pretrained word vectors to find words similar to the search word, and search for those as well. For example, we can find words related to "weapons" by running the python code.

```
>>> import gensim.downloader
>>> vectors = gensim.downloader.load('glove-twitter-25')
>>> vectors.most_similar('weapons')
[('drones', 0.8980589509010315),
 ('drone', 0.8965809345245361),
 ('assault', 0.8937929272651672),
 ('targets', 0.8834593296051025),
 ('firearms', 0.8833326697349548),
 ('weapon', 0.8730441927909851),
 ('hiv', 0.8625047206878662),
 ('laws', 0.8613511323928833),
 ('drug', 0.8555657863616943),
 ('concealed', 0.8548306822776794)
]
```
Notice that both the words "drones" and "targets" appear in this list of similar words. The list isn't perfect though... "hiv" doesn't seem very similar to "weapons" to me.

Part of the problem is that I'm using a particularly poor model above. The glove-twitter-25 model is trained only on twitter data and has only 25 dimensions. Larger models trained on more data naturally provide better results.

You can find a list of models built-in to gensim here, and there's thousands of other open source models that people have released that can easily be incorporated as well.

**Your Task**: Modify the `pagerank.py` file so that it also searches for the keywords in the query and the 5 most similar words. The results of your modified file when searching for "weapons" should look something like:
```
$ python3 pagerank.py --data=./lawfareblog.csv.gz --search_query='weapons'
INFO:root:rank=0 pagerank=0.004571518860757351 url=www.lawfareblog.com/why-did-you-wait-moral-emptiness-and-drone-strikes
INFO:root:rank=1 pagerank=0.0031107424292713404 url=www.lawfareblog.com/dc-district-court-dismisses-journalists-drone-lawsuit
INFO:root:rank=2 pagerank=0.0020231129601597786 url=www.lawfareblog.com/revived-cia-drone-strike-program-comments-new-policy
INFO:root:rank=3 pagerank=0.0019667143933475018 url=www.lawfareblog.com/us-court-appeals-dc-circuit-dismisses-suit-over-us-drone-strike
INFO:root:rank=4 pagerank=0.001178761012852192 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=5 pagerank=0.0011619674041867256 url=www.lawfareblog.com/slaughterbots-and-other-anticipated-autonomous-weapons-problems
INFO:root:rank=6 pagerank=0.0011276121949777007 url=www.lawfareblog.com/german-courts-weigh-legal-responsibility-us-drone-strikes
INFO:root:rank=7 pagerank=0.0008373793680220842 url=www.lawfareblog.com/shift-jsoc-drone-strikes-does-not-mean-cia-has-been-sidelined
INFO:root:rank=8 pagerank=0.0007856971933506429 url=www.lawfareblog.com/waiving-imminent-threat-test-cia-drone-strikes-pakistan
INFO:root:rank=9 pagerank=0.0007412837585434318 url=www.lawfareblog.com/drone-strike-errors-and-hostage-tragedy-mapping-issues-newly-catalyzed-debate
```
Notice that most of these articles do not mention the word "weapons", but instead mention the word "drone".

**Submission**: Upload your completed pagerank.py to github, and update the README.md file from your pagerank homework assignment so that all the displayed results are generated from your modified search engine. (You don't need to redo the results for the small.csv.gz file, just the lawfareblog.csv.gz file.)
