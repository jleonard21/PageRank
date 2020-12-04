# Pagerank HW: Web Search with Word2Vec

In this assignment, we create a search engine that uses both pagerank and word embeddings to find high quality results.

**Due date:** Sunday, 6 December at midnight

## Task 1: Power Method
Implement the WebGraph.power_method function in `pagerank2.py` for computing the pagerank vector.

**Part 1:** To check that the implementation is working, run the program on the `small.csv.gz` graph:
```
In [1]: run pagerank2.py --data=./small.csv.gz --verbose
DEBUG:root:i=0 accuracy=tensor(0.2563) 
DEBUG:root:i=1 accuracy=tensor(0.1184) 
DEBUG:root:i=2 accuracy=tensor(0.0707) 
DEBUG:root:i=3 accuracy=tensor(0.0318) 
DEBUG:root:i=4 accuracy=tensor(0.0205) 
DEBUG:root:i=5 accuracy=tensor(0.0101) 
DEBUG:root:i=6 accuracy=tensor(0.0064) 
DEBUG:root:i=7 accuracy=tensor(0.0034) 
DEBUG:root:i=8 accuracy=tensor(0.0021) 
DEBUG:root:i=9 accuracy=tensor(0.0012) 
DEBUG:root:i=10 accuracy=tensor(0.0007)
DEBUG:root:i=11 accuracy=tensor(0.0004)
DEBUG:root:i=12 accuracy=tensor(0.0002)
DEBUG:root:i=13 accuracy=tensor(0.0001)
DEBUG:root:i=14 accuracy=tensor(8.1083e-05)
DEBUG:root:i=15 accuracy=tensor(4.7251e-05)
DEBUG:root:i=16 accuracy=tensor(2.7704e-05)
DEBUG:root:i=17 accuracy=tensor(1.6164e-05)
DEBUG:root:i=18 accuracy=tensor(9.4778e-06)
DEBUG:root:i=19 accuracy=tensor(5.5066e-06)
DEBUG:root:i=20 accuracy=tensor(3.2042e-06)
DEBUG:root:i=21 accuracy=tensor(1.8612e-06)
DEBUG:root:i=22 accuracy=tensor(1.1283e-06)
DEBUG:root:i=23 accuracy=tensor(6.1907e-07)
INFO:root:rank=0 pagerank=0.66269850730896 url=4
INFO:root:rank=1 pagerank=0.5217869281768799 url=6
INFO:root:rank=2 pagerank=0.4143447279930115 url=5
INFO:root:rank=3 pagerank=0.2317543923854828 url=2
INFO:root:rank=4 pagerank=0.18590237200260162 url=3
INFO:root:rank=5 pagerank=0.16916769742965698 url=1
```

**Part 2:** The `pagerank2.py` file has an option `--search_query`, which takes a string as a parameter. If this argument is used, then program returns all urls that match the query string sorted according to their pagerank. Essentially, this gives us the most important pages on the blog related to our query:
```
In [2]: run pagerank2.py --data=./lawfareblog.csv.gz --search_query='corona'                     
INFO:root:rank=0 pagerank=0.001003776676952839 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=0.0008922395063564181 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=0.0007039029151201248 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=0.0006915341946296394 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=0.000670412031468004 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=0.0006625585374422371 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=0.0006504578050225973 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=0.0006361958803609014 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=0.000612482544966042 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=0.0006018723943270743 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

In [3]: run pagerank2.py --data=./lawfareblog.csv.gz --search_query='trump'                      
INFO:root:rank=0 pagerank=0.005782557651400566 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=0.005233839154243469 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=0.005129670724272728 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=0.004659898113459349 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=0.004593398422002792 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=0.004307133611291647 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=0.0040934765711426735 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=0.0037590833380818367 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=0.003450872143730521 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=0.0034484383650124073 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

In [4]: run pagerank2.py --data=./lawfareblog.csv.gz --search_query='iran'                       
INFO:root:rank=0 pagerank=0.005129670724272728 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=1 pagerank=0.005016769748181105 url=www.lawfareblog.com/update-military-commissions-continued-health-issues-recusal-motion-and-new-cell-al-iraqi
INFO:root:rank=2 pagerank=0.004574609454721212 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=3 pagerank=0.004417411983013153 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=4 pagerank=0.0034236714709550142 url=www.lawfareblog.com/france-makes-play-try-foreign-fighters-iraq
INFO:root:rank=5 pagerank=0.0026927897706627846 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=6 pagerank=0.002256679581478238 url=www.lawfareblog.com/document-sens-kaine-and-young-introduce-bill-revoke-iraq-force-authorizations
INFO:root:rank=7 pagerank=0.0019391420064494014 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=8 pagerank=0.0018262730445712805 url=www.lawfareblog.com/its-not-only-iraq-and-syria
INFO:root:rank=9 pagerank=0.001733092125505209 url=www.lawfareblog.com/assessing-aclu-habeas-petition-behalf-unnamed-us-citizen-held-enemy-combatant-iraq
```

**Part 3:** The webgraph of lawfareblog.com (the P matrix) naturally contains a lot of structure. For example, essentially all pages on the domain have links to the root page https://lawfareblog.com/ and similarly broad pages like https://www.lawfareblog.com/topics and https://www.lawfareblog.com/subscribe-lawfare. These pages therefore have a large pagerank. We can get a list of the pages with the largest pagerank by running
```
In [5]: run pagerank2.py --data=./lawfareblog.csv.gz                                             
INFO:root:rank=0 pagerank=0.2874051630496979 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=1 pagerank=0.2874051630496979 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=0.2874051630496979 url=www.lawfareblog.com/masthead
INFO:root:rank=3 pagerank=0.2874051630496979 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=0.2874051630496979 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=0.2874051630496979 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=6 pagerank=0.2874051630496979 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=7 pagerank=0.2874051630496979 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=0.2874051630496979 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=0.2874051630496979 url=www.lawfareblog.com/topics
```

These pages are not very interesting, however, because they are not articles. To find the most important articles we modify the P matrix by removing all links to non-article pages.

An easy method is to remove all pages that have "too many" links. The --filter_ratio argument removes all pages that have more links than the specified fraction.

We can estimate the most important articles on the domain with the `filter ratio = 0.2`:
```
In [6]: run pagerank2.py --data=./lawfareblog.csv.gz --filter_ratio=0.2                          
INFO:root:rank=0 pagerank=0.3469613492488861 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=0.29521211981773376 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=0.29039666056632996 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=0.15178653597831726 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=0.15098513662815094 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=5 pagerank=0.15098513662815094 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=6 pagerank=0.15071173012256622 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=0.14956679940223694 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=0.14366623759269714 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=0.14239734411239624 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```

**Part 4:** 
Run the following code:
```
In [7]: run pagerank2.py --data=./lawfareblog.csv.gz --verbose 
In [8]: run pagerank2.py --data=./lawfareblog.csv.gz --verbose --alpha=0.99999
In [9]: run pagerank2.py --data=./lawfareblog.csv.gz --verbose --filter_ratio=0.2
In [10]: run pagerank2.py --data=./lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
Changing the value of alpha also gives us very different pagerank rankings. For example:
```
In [11]: run pagerank2.py --data=./lawfareblog.csv.gz --verbose                                   
INFO:root:rank=0 pagerank=0.2874051630496979 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=1 pagerank=0.2874051630496979 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=0.2874051630496979 url=www.lawfareblog.com/masthead
INFO:root:rank=3 pagerank=0.2874051630496979 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=0.2874051630496979 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=0.2874051630496979 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=6 pagerank=0.2874051630496979 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=7 pagerank=0.2874051630496979 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=0.2874051630496979 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=0.2874051630496979 url=www.lawfareblog.com/topics 

In [12]: run pagerank2.py --data=./lawfareblog.csv.gz --verbose --alpha=0.99999                   
INFO:root:rank=0 pagerank=0.28859302401542664 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=1 pagerank=0.28859302401542664 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=0.28859302401542664 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=3 pagerank=0.28859302401542664 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=0.28859302401542664 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=0.28859302401542664 url=www.lawfareblog.com/topics
INFO:root:rank=6 pagerank=0.28859302401542664 url=www.lawfareblog.com/masthead
INFO:root:rank=7 pagerank=0.28859302401542664 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=0.28859302401542664 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=0.28859302401542664 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general

In [13]: run pagerank2.py --data=./lawfareblog.csv.gz --verbose --filter_ratio=0.2                
INFO:root:rank=0 pagerank=0.3469613492488861 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=0.29521211981773376 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=0.29039666056632996 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=0.15178653597831726 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=0.15098513662815094 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=5 pagerank=0.15098513662815094 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=6 pagerank=0.15071173012256622 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=0.14956679940223694 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=0.14366623759269714 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=0.14239734411239624 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

In [14]: run pagerank2.py --data=./lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=0.7014895677566528 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=0.7014873623847961 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=0.10551629960536957 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=0.03175504133105278 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=0.022039713338017464 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=0.01602667197585106 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=0.016025930643081665 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=0.016022762283682823 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=0.016019931063055992 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=0.016019923612475395 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

### Task 2: Personalization Vector
Recall that we can use this file to find the "highest quality" search results from the https://lawfareblog.com website.

**Part 1:**  Implement the WebGraph.make_personalization_vector function. This function enables the --personalization_vector_query command line argument, which provides an alternative method for searching by doing the filtering on the personalization vector.
```
In [15]: run pagerank2.py --data=./lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=0.6320855617523193 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=0.6320620179176331 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=0.15656693279743195 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=0.12040437757968903 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=4 pagerank=0.12040437757968903 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=5 pagerank=0.09199091792106628 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=0.08998079597949982 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=0.08998079597949982 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=0.07601435482501984 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=0.0717419981956482 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

However, these results are different than when using the --search_query option:
```
In [16]: run pagerank2.py --data=./lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona' 
INFO:root:rank=0 pagerank=0.008131962269544601 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=0.007790825795382261 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=0.005226220469921827 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=0.0039583845064044 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=0.003811446251347661 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=0.003397284774109721 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=0.0033633282873779535 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=0.0033556800335645676 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=0.0032159897964447737 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=0.0031036492437124252 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

**Part 2:** Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus. This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance, but removes webpages mentioning `corona` from the results:
```
In [17]: run pagerank2.py --data=./lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=0.6320855617523193 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=0.6320620179176331 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=0.15656693279743195 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=0.09199091792106628 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=0.06943929940462112 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=5 pagerank=0.06895698606967926 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=6 pagerank=0.06389345228672028 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=0.05843931436538696 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=0.0507182851433754 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=0.05071817338466644 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```

**Part 3:** You should experiment with a national security topic other than the coronavirus. For example, find out what articles are important to the `iran` topic but do not contain the word `iran`. Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
```
In [18]: run pagerank2.py --data=./lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='iran' --search_query='-iran'
```
