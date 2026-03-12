# MCQ Generation: Categories + Prompt

## Prompt Template

**System prompt:**

> You are a trivia question generator. You create factually accurate multiple-choice questions about U.S. presidents. Each question is one short sentence. Each question has exactly 4 options labeled A, B, C, D. Options are short (1-5 words each). Exactly one option is correct. The 3 wrong options should be plausible but clearly wrong. Randomize the position of the correct answer. Respond ONLY with valid JSON matching this schema, no markdown, no preamble:
>
> {"questions": [{"question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_answer": "A"}]}

**User prompt:**

> Generate 20 unique multiple-choice trivia questions about {president} specifically related to: {category}.

---

## The Math

- 5,000 questions per president
- 20 questions per API call
- 250 calls per president
- 250 categories per president (1 round each)
- 3 presidents × 250 = 750 total API calls
- 15,000 total questions

---

## Categories (250)

### Early Life (1-15)

1. Birthplace and birth date
2. Childhood neighborhood
3. Mother's background
4. Father's background
5. Siblings
6. Childhood hobbies
7. Elementary school years
8. Middle school years
9. High school years
10. Childhood friends
11. Childhood homes and moves
12. Family religion during childhood
13. Childhood pets
14. Grandparents
15. Childhood hardships

### Education (16-30)

16. Undergraduate college choice
17. College major
18. College extracurricular activities
19. College grades and academic performance
20. Graduate school
21. Law school
22. Professors and academic mentors
23. College thesis or notable papers
24. Scholarships and financial aid
25. Study abroad experiences
26. College friendships
27. College athletics
28. Graduation year and details
29. Influence of education on political views
30. Honorary degrees received

### Family (31-50)

31. How they met their spouse
32. Wedding details
33. Spouse's childhood and background
34. Spouse's education
35. Spouse's career before the White House
36. Children's names and birth years
37. Children's education
38. Children's careers
39. Grandchildren
40. Relationship with parents during adulthood
41. Extended family members in public life
42. Family vacations
43. Pets in the White House
44. Family holiday traditions
45. Spouse's role during campaigns
46. Spouse's White House initiatives
47. Children's lives during the presidency
48. Family homes and real estate
49. Family controversies
50. Family's post-presidency life

### Pre-Political Career (51-70)

51. First job
52. Legal career
53. Community organizing work
54. Business ventures
55. Real estate career
56. Teaching career
57. Television career
58. Writing career before presidency
59. Military service
60. Board memberships
61. Nonprofit involvement
62. Mentors in early career
63. Key colleagues before politics
64. Financial situation before politics
65. Volunteer work
66. Career pivots
67. Professional awards before politics
68. Published articles before politics
69. Public speaking before politics
70. Influential experiences before politics

### Entry into Politics (71-85)

71. First political campaign
72. First elected office
73. Early political mentors
74. Early political allies
75. Party affiliation history
76. First political speech
77. Local government involvement
78. State-level political career
79. U.S. Senate career
80. Congressional voting record
81. Political fundraising beginnings
82. Early political controversies
83. Key endorsements received early on
84. Political ideology development
85. Grassroots organizing

### Presidential Campaign (86-110)

86. Campaign announcement details
87. Primary opponents
88. Primary debate moments
89. Campaign slogan
90. Campaign manager
91. Campaign fundraising totals
92. Key campaign staff
93. Campaign trail events
94. Campaign advertising
95. Social media campaign strategy
96. Celebrity endorsements
97. Union endorsements
98. Newspaper endorsements
99. Campaign gaffes
100. Campaign controversies
101. Debate performance against opponent
102. Vice presidential selection process
103. Running mate announcement
104. Convention speech
105. Campaign bus or plane details
106. Swing state strategy
107. Campaign rallies
108. Opposition research attacks
109. Campaign promises
110. Campaign volunteers and ground game

### Presidential Election (111-125)

111. Election night details
112. Electoral college vote count
113. Popular vote total
114. Key states won
115. Key states lost
116. Voter demographics breakdown
117. Opponent's concession
118. Victory speech
119. Margin of victory
120. Turnout statistics
121. Third party candidates in the race
122. Recounts or election disputes
123. International reaction to the election
124. Stock market reaction to the election
125. Historical significance of the election

### Inauguration (126-135)

126. Inauguration date
127. Inauguration weather
128. Inauguration speech themes
129. Who administered the oath
130. Bible used for swearing in
131. Inauguration performers
132. Inauguration attendance
133. Inaugural parade details
134. Inaugural balls
135. First actions after inauguration

### White House Life (136-150)

136. Daily routine as president
137. Favorite foods in the White House
138. Exercise habits as president
139. White House staff favorites
140. Oval Office decor choices
141. Camp David visits
142. Air Force One details
143. Presidential limousine
144. White House entertaining style
145. State dinners hosted
146. White House renovations
147. Vacation spots during presidency
148. Hobbies during presidency
149. Books read during presidency
150. Music taste during presidency

### Domestic Policy (151-175)

151. Job creation record
152. Unemployment rate during presidency
153. GDP growth during presidency
154. Tax policy changes
155. Minimum wage positions
156. Infrastructure spending
157. Housing policy
158. Education policy
159. Student loan policy
160. Gun control positions
161. Opioid crisis response
162. Criminal justice reform
163. Prison reform
164. Policing policy
165. Drug policy
166. Veteran affairs policy
167. Social Security positions
168. Medicare positions
169. Medicaid expansion
170. Welfare policy
171. Small business policy
172. Technology policy
173. Cybersecurity policy
174. Rural development policy
175. Urban development policy

### Healthcare (176-185)

176. Signature healthcare legislation
177. Healthcare executive orders
178. Prescription drug pricing
179. Mental health policy
180. Obamacare (ACA) positions
181. Medicare for All positions
182. Health insurance coverage changes
183. Hospital policy
184. Pandemic preparedness
185. Public health initiatives

### Economic Policy (186-200)

186. Stock market performance
187. National debt changes
188. Federal budget proposals
189. Stimulus packages
190. Trade deficit changes
191. Manufacturing policy
192. Banking regulation
193. Wall Street regulation
194. Federal Reserve relations
195. Inflation during presidency
196. Interest rate environment
197. Corporate tax changes
198. Capital gains tax positions
199. Estate tax positions
200. Economic advisors

### Immigration (201-210)

201. Border wall positions
202. DACA policy
203. Refugee admission numbers
204. Travel ban policies
205. Deportation statistics
206. Immigration executive orders
207. Dreamers policy
208. Family separation policy
209. Legal immigration changes
210. Asylum policy changes

### Foreign Policy (211-235)

211. Relations with China
212. Relations with Russia
213. Relations with North Korea
214. Relations with Iran
215. Relations with Israel
216. Relations with NATO allies
217. Relations with the United Kingdom
218. Relations with Mexico
219. Relations with Canada
220. Relations with Saudi Arabia
221. Relations with European Union
222. Middle East peace efforts
223. Africa policy
224. Latin America policy
225. Asia Pacific strategy
226. United Nations positions
227. International agreements signed
228. International agreements withdrawn from
229. State visits received
230. State visits made abroad
231. G7 summit participation
232. G20 summit participation
233. Diplomatic achievements
234. Diplomatic failures
235. Embassy decisions

### Military and Defense (236-250)

236. Military spending levels
237. Troop deployment decisions
238. Troop withdrawal decisions
239. Afghanistan policy
240. Iraq policy
241. Syria policy
242. Drone strike policy
243. Special operations missions
244. Military leadership appointments
245. Defense secretary choices
246. Nuclear weapons policy
247. Arms deals with foreign nations
248. Military base decisions
249. Counterterrorism strategy
250. Veterans policy

### Supreme Court and Judiciary (251-260)

251. Supreme Court nominations
252. Supreme Court confirmation battles
253. Federal judge appointments total
254. Circuit court appointments
255. District court appointments
256. Judicial philosophy preferences
257. Landmark Supreme Court cases during term
258. Attorney General selections
259. Department of Justice controversies
260. Solicitor General positions

### Executive Power (261-275)

261. Executive orders signed total
262. Most significant executive orders
263. Presidential memoranda
264. Pardons granted
265. Commutations granted
266. Vetoes issued
267. Pocket vetoes
268. Signing statements
269. Emergency declarations
270. Use of executive privilege
271. Recess appointments
272. Presidential proclamations
273. Cabinet firings
274. White House staff turnover
275. Presidential task forces created

### Environment and Energy (276-290)

276. Paris Climate Agreement positions
277. EPA regulation changes
278. Clean energy investment
279. Fossil fuel policy
280. Oil drilling policy
281. Pipeline decisions
282. Fuel efficiency standards
283. Carbon emission targets
284. National park and monument decisions
285. Endangered species protections
286. Water regulation policy
287. Air quality standards
288. Nuclear energy positions
289. Electric vehicle policy
290. Environmental justice initiatives

### Trade Policy (291-300)

291. NAFTA and USMCA positions
292. Trans-Pacific Partnership positions
293. China tariffs
294. Steel and aluminum tariffs
295. European Union trade disputes
296. World Trade Organization positions
297. Trade agreements signed
298. Trade war impacts
299. Export policy changes
300. Import policy changes

### Technology and Science (301-310)

301. Space policy and NASA
302. Big tech regulation positions
303. Social media policy
304. Artificial intelligence policy
305. Net neutrality positions
306. Data privacy positions
307. 5G and telecom policy
308. Science advisor appointments
309. Research funding levels
310. STEM education initiatives

### COVID-19 (311-325)

311. Initial pandemic response
312. Travel restrictions during COVID
313. Mask mandate positions
314. Vaccine development efforts
315. Vaccine distribution strategy
316. Economic relief packages
317. Small business pandemic aid
318. School closure positions
319. COVID testing strategy
320. COVID death toll during tenure
321. Relations with WHO during COVID
322. CDC guidance during presidency
323. Pandemic press briefings
324. State vs federal COVID conflicts
325. COVID economic impact during tenure

### Scandals and Controversies (326-345)

326. Impeachment proceedings
327. Impeachment charges
328. Impeachment vote results
329. Special counsel investigations
330. FBI investigations
331. Congressional investigations
332. Ethics complaints
333. Controversial pardons
334. Classified document controversies
335. Personal conduct controversies
336. Financial disclosure controversies
337. Conflict of interest allegations
338. Campaign finance controversies
339. Social media controversies
340. Media feuds
341. Controversial statements
342. Staff controversies
343. Family member controversies
344. Business-related controversies
345. Legal challenges during presidency

### Congressional Relations (346-360)

346. Relationship with House Speaker
347. Relationship with Senate Majority Leader
348. Government shutdown involvement
349. Filibuster positions
350. Bipartisan legislation achievements
351. Key legislation signed
352. Key legislation vetoed
353. Debt ceiling negotiations
354. Budget battles
355. Congressional approval of nominees
356. State of the Union addresses
357. Joint sessions of Congress
358. Midterm election impacts
359. Congressional majority during term
360. Lame duck session actions

### Key Speeches and Communication (361-375)

361. Inaugural address memorable lines
362. State of the Union memorable moments
363. Press conference style
364. Major national addresses
365. Speeches at memorials
366. Commencement speeches
367. UN General Assembly speeches
368. Nobel Prize speech (if applicable)
369. Campaign victory speech
370. Farewell address
371. Twitter usage as president
372. Relationship with White House press corps
373. Ghostwriters and speechwriters
374. Catchphrases and verbal habits
375. Communication strategy

### Cabinet and Administration (376-395)

376. Secretary of State choices
377. Secretary of Defense choices
378. Secretary of Treasury choices
379. Attorney General choices
380. National Security Advisor choices
381. Chief of Staff choices
382. CIA Director choices
383. FBI Director choices
384. UN Ambassador choices
385. Press Secretary choices
386. Cabinet diversity
387. Cabinet confirmations battles
388. Cabinet resignations
389. Acting officials in cabinet roles
390. Senior advisor appointments
391. White House Counsel choices
392. OMB Director choices
393. Trade Representative choices
394. EPA Administrator choices
395. Surgeon General choices

### Civil Rights and Social Issues (396-415)

396. LGBTQ rights positions
397. Same-sex marriage positions
398. Racial justice positions
399. Police reform positions
400. Affirmative action positions
401. Voting rights positions
402. Women's rights positions
403. Abortion policy positions
404. Religious freedom positions
405. Hate crime legislation
406. Disability rights positions
407. Native American policy
408. Asian American policy responses
409. Hispanic community outreach
410. Black community relations
411. Antisemitism responses
412. Islamophobia responses
413. Protest responses during presidency
414. Civil liberties positions
415. Census policy decisions

### Elections and Political Strategy (416-430)

416. Re-election campaign details
417. Primary challenges faced
418. Debate strategies
419. Key campaign advisors
420. Super PAC involvement
421. Donor demographics
422. Battleground state results
423. Coattail effect on other races
424. Party platform influence
425. Opposition party's candidate
426. Third party impact on election
427. Early voting impact
428. Mail-in voting positions
429. Voter fraud claims
430. Election night timeline

### Media and Public Image (431-445)

431. Approval ratings over time
432. Highest approval rating moment
433. Lowest approval rating moment
434. Relationship with Fox News
435. Relationship with CNN
436. Relationship with New York Times
437. Late night TV appearances
438. Saturday Night Live portrayals
439. Documentary films about them
440. Biographical books about them
441. Books written by them
442. Magazine covers
443. Photographer relationships
444. Memes and viral moments
445. Public perception polls

### Post-Presidency (446-460)

446. Post-presidency residence
447. Presidential library details
448. Post-presidency book deals
449. Post-presidency speaking fees
450. Post-presidency political involvement
451. Post-presidency endorsements
452. Charitable work after office
453. Post-presidency travels
454. Post-presidency media appearances
455. Relationship with successor
456. Post-presidency controversies
457. Post-presidency business ventures
458. Post-presidency awards received
459. Post-presidency public statements
460. Post-presidency approval ratings

### Legacy and Historical Ranking (461-475)

461. Historians' ranking among presidents
462. Signature policy legacy
463. Impact on the political party
464. Impact on future elections
465. Most remembered moment
466. Most criticized decision
467. Most praised decision
468. Comparison to predecessor
469. Comparison to successor
470. Cultural impact
471. Impact on political polarization
472. Impact on media landscape
473. Long-term economic legacy
474. Long-term foreign policy legacy
475. Long-term judicial legacy

### Specific Events During Presidency (476-510)

476. Natural disasters during presidency
477. Mass shootings during presidency
478. Terrorist attacks during presidency
479. Major protests during presidency
480. Stock market crashes during presidency
481. Government shutdowns during presidency
482. International crises during presidency
483. Diplomatic incidents during presidency
484. Military conflicts started
485. Military conflicts ended
486. Peace agreements brokered
487. Hostage situations
488. Cyberattacks during presidency
489. Public health emergencies
490. Infrastructure failures during presidency
491. Space achievements during presidency
492. Major scientific discoveries during term
493. Census results during presidency
494. Olympic Games during presidency
495. World Cup during presidency
496. Major legislation failures
497. Vetoed legislation
498. Executive order reversals by successor
499. Controversial Cabinet meetings
500. Secret Service incidents

### Awards and Honors (501-510)

501. Nobel Prize
502. Grammy Awards
503. Emmy nominations or wins
504. Time Person of the Year
505. Foreign honors received
506. State honors received
507. Academic honors
508. Sports-related honors
509. Cultural awards
510. Humanitarian awards

### Relationships with World Leaders (511-525)

511. Relationship with UK Prime Minister
512. Relationship with German Chancellor
513. Relationship with French President
514. Relationship with Japanese Prime Minister
515. Relationship with Chinese President
516. Relationship with Russian President
517. Relationship with Indian Prime Minister
518. Relationship with Australian Prime Minister
519. Relationship with South Korean President
520. Relationship with Brazilian President
521. Relationship with Canadian Prime Minister
522. Relationship with Mexican President
523. Relationship with Turkish President
524. Relationship with Egyptian President
525. Relationship with Pope

### Personal Characteristics (526-540)

526. Height and physical description
527. Handedness
528. Favorite sports
529. Favorite sports teams
530. Favorite movies
531. Favorite TV shows
532. Favorite musicians
533. Religious faith
534. Health issues during presidency
535. Exercise routine
536. Diet preferences
537. Fashion choices
538. Sense of humor
539. Nicknames
540. Personal financial net worth

### Political Philosophy (541-555)

541. Self-described political ideology
542. Influences on political thinking
543. Key policy differences from predecessor
544. Key policy differences from successor
545. Evolution of political views
546. Stance on government spending
547. Stance on federal vs state power
548. Stance on regulation
549. Stance on free trade vs protectionism
550. Stance on interventionism vs isolationism
551. Stance on entitlement programs
552. Stance on campaign finance reform
553. Stance on term limits
554. Stance on Electoral College
555. Stance on constitutional amendments

### Transition of Power (556-565)

556. Transition team members
557. Transition controversies
558. Lame duck actions of predecessor
559. Day one executive actions
560. First Cabinet meeting
561. First foreign leader call
562. First foreign leader visit
563. First 100 days achievements
564. First 100 days failures
565. Transition budget and logistics

### State-Level Impact (566-575)

566. Home state political influence
567. Governor relationships
568. State election impacts
569. State policy influences
570. State party leadership changes
571. Home state approval ratings
572. State-federal conflicts
573. Natural disaster state responses
574. State economic impacts
575. State-level political allies

### Vice Presidency (576-585)

576. Vice president selection criteria
577. Vice president's background
578. Vice president's role in administration
579. Vice president's key assignments
580. Vice president's public gaffes
581. Vice president's relationship with president
582. Vice president's presidential ambitions
583. Vice president's debate performance
584. Vice president's tie-breaking Senate votes
585. Vice president's foreign trips

### Firsts and Records (586-600)

586. Historical firsts achieved
587. Records broken during presidency
588. Youngest or oldest records
589. Longest or shortest records
590. First president to do something specific
591. Electoral records
592. Legislative records
593. Executive order records
594. Appointment records
595. Travel records as president
596. Fundraising records
597. Social media records
598. Approval rating records
599. Economic records during term
600. Diplomatic firsts

### Intelligence and National Security (601-615)

601. Intelligence briefing habits
602. NSA surveillance positions
603. CIA operations approved
604. Whistleblower responses
605. Classified information handling
606. National security strategy documents
607. Homeland Security policy
608. Border security spending
609. Cybersecurity executive orders
610. Election security measures
611. Foreign interference responses
612. Intelligence community relations
613. National Security Council structure
614. Situation Room decisions
615. Presidential daily briefing format

### Infrastructure and Transportation (616-625)

616. Highway and road spending
617. Bridge and dam investment
618. Public transit policy
619. Airport modernization
620. Broadband expansion
621. Water infrastructure
622. Power grid policy
623. Rail policy
624. Port investment
625. Infrastructure legislation

### Agriculture and Rural Policy (626-635)

626. Farm bill positions
627. Agricultural subsidy policy
628. Rural broadband initiatives
629. Farming trade impacts
630. Food safety regulation
631. USDA leadership
632. Ethanol and biofuel policy
633. Crop insurance policy
634. Rural healthcare access
635. Agricultural tariff impacts

### Housing and Urban Policy (636-645)

636. Affordable housing initiatives
637. HUD Secretary appointments
638. Homelessness policy
639. Mortgage regulation
640. Rent control positions
641. Public housing investment
642. Urban revitalization programs
643. Opportunity zones
644. Fair housing enforcement
645. Foreclosure prevention

### Labor and Workforce (646-655)

646. Union relationship
647. Right-to-work positions
648. Overtime rule changes
649. Workplace safety regulation
650. Gig economy policy
651. Federal employee pay
652. Government hiring freezes
653. Workforce training programs
654. Apprenticeship initiatives
655. Labor Department leadership

### Financial Regulation (656-665)

656. Dodd-Frank positions
657. Consumer Financial Protection Bureau
658. Bank bailout positions
659. Cryptocurrency positions
660. SEC leadership
661. Federal Reserve chair appointments
662. Too big to fail positions
663. Credit card regulation
664. Mortgage lending rules
665. Financial crisis responses

### Tribal and Indigenous Affairs (666-670)

666. Tribal sovereignty positions
667. Pipeline disputes with tribes
668. Sacred land protections
669. Bureau of Indian Affairs policy
670. Tribal consultation policies

### Space Policy (671-680)

671. NASA budget decisions
672. Space exploration goals
673. Mars mission positions
674. Moon mission positions
675. Space Force creation
676. Commercial space partnerships
677. International Space Station policy
678. Satellite and debris policy
679. Space science funding
680. Astronaut milestones during term

### Disaster Response (681-690)

681. Hurricane response record
682. Wildfire response
683. Earthquake responses
684. Flood response
685. Tornado response
686. FEMA leadership
687. Disaster declaration totals
688. Emergency funding requests
689. Climate disaster responses
690. Industrial disaster responses

### Voting and Elections Policy (691-700)

691. Voting Rights Act positions
692. Voter ID law positions
693. Gerrymandering positions
694. Election security funding
695. Campaign finance law positions
696. Citizens United positions
697. Early voting expansion
698. Felony disenfranchisement positions
699. DC statehood positions
700. Puerto Rico statehood positions

### International Trade Agreements (701-710)

701. Bilateral trade deals signed
702. Multilateral trade positions
703. Sanctions imposed
704. Sanctions lifted
705. Export control policy
706. Import tariff specifics
707. Trade representative achievements
708. WTO dispute involvement
709. Trade with Africa
710. Trade with Southeast Asia

### Education Policy Details (711-720)

711. K-12 education reform
712. Common Core positions
713. Charter school positions
714. School choice positions
715. Teacher pay positions
716. College affordability initiatives
717. For-profit college regulation
718. STEM funding
719. Title IX policy
720. School safety initiatives

### Drug Policy (721-730)

721. Marijuana legalization positions
722. Opioid epidemic response
723. Drug scheduling decisions
724. DEA leadership
725. Drug trafficking enforcement
726. Needle exchange positions
727. Drug treatment funding
728. Fentanyl crisis response
729. Prescription monitoring
730. International drug policy

### Social Media and Technology Impact (731-740)

731. Personal social media usage
732. Campaign social media strategy
733. Social media bans or restrictions
734. Section 230 positions
735. Misinformation response
736. Platform moderation positions
737. Digital advertising in campaigns
738. Online fundraising innovations
739. Viral moments on social media
740. Social media following numbers

### Arts and Culture (741-750)

741. White House cultural events
742. National arts funding
743. Kennedy Center Honors during term
744. Presidential Medal of Freedom recipients
745. White House music performances
746. Support for public broadcasting
747. NEA and NEH funding
748. Cultural diplomacy efforts
749. Historical preservation decisions
750. Museum and monument dedications
