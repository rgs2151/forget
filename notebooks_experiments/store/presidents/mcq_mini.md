# MCQ Generation: 250 Categories + Prompt

## Prompt

**System prompt:**

> You are a trivia question generator. You create factually accurate multiple-choice questions about U.S. presidents. Each question is one short sentence. Each question has exactly 4 options labeled A, B, C, D. Options are short (1-5 words each). Exactly one option is correct. The 3 wrong options should be plausible but clearly wrong. Randomize the position of the correct answer. Respond ONLY with valid JSON matching this schema, no markdown, no preamble:
>
> {"questions": [{"question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_answer": "A"}]}

**User prompt:**

> Generate 20 unique multiple-choice trivia questions about {president} specifically related to: {category}.

---

## 250 Categories

1. Birthplace and birth date
2. Childhood neighborhood and hometown
3. Mother's background and career
4. Father's background and career
5. Siblings and sibling relationships
6. Childhood hobbies and interests
7. Elementary school experiences
8. High school years
9. Grandparents
10. Childhood hardships and challenges
11. Undergraduate college and major
12. College extracurricular activities
13. Graduate or professional school
14. Professors and academic mentors
15. College friendships and social life
16. Graduation year and academic record
17. How they met their spouse
18. Wedding details
19. Spouse's childhood and education
20. Spouse's career before the White House
21. Children's names and birth years
22. Children's education and schools
23. Children's careers and public lives
24. Pets in the White House
25. Family holiday traditions
26. Spouse's White House initiatives
27. Children's lives during the presidency
28. Family homes and real estate
29. Family controversies
30. Family's post-presidency life
31. First job after school
32. Legal career
33. Community organizing or volunteer work
34. Business ventures and investments
35. Teaching or academic career
36. Writing career before presidency
37. Board memberships and nonprofit work
38. Mentors in early career
39. Key colleagues before politics
40. Financial situation before politics
41. First political campaign
42. First elected office held
43. Early political mentors
44. Early political allies
45. Party affiliation history
46. State-level political career
47. U.S. Senate career
48. Congressional voting record
49. Early political controversies
50. Grassroots organizing history
51. Presidential campaign announcement
52. Primary election opponents
53. Primary debate moments
54. Campaign slogan
55. Campaign manager and key staff
56. Campaign fundraising totals
57. Celebrity endorsements received
58. Union endorsements received
59. Newspaper endorsements received
60. Campaign gaffes and mistakes
61. Campaign trail controversies
62. General election debate performance
63. Vice presidential selection process
64. Running mate announcement
65. Convention speech highlights
66. Swing state strategy
67. Campaign rallies and events
68. Opposition attacks faced
69. Campaign promises made
70. Campaign volunteer and ground game
71. Election night events
72. Electoral college vote count
73. Popular vote total
74. Key states won
75. Key states lost
76. Voter demographics breakdown
77. Opponent's concession speech
78. Victory speech details
79. Margin of victory
80. Turnout statistics
81. Third party candidates in the race
82. International reaction to the election
83. Stock market reaction to the election
84. Historical significance of the election
85. Inauguration date and weather
86. Inauguration speech themes
87. Who administered the oath
88. Inauguration performers and entertainment
89. Inauguration attendance estimates
90. Inaugural parade details
91. Inaugural balls
92. First actions after inauguration
93. Daily routine as president
94. Favorite foods in the White House
95. Exercise habits as president
96. Oval Office decor choices
97. Camp David visits
98. Air Force One details
99. White House entertaining and state dinners
100. Vacation spots during presidency
101. Hobbies during presidency
102. Books read or recommended as president
103. Job creation record
104. Unemployment rate during presidency
105. GDP growth during presidency
106. Tax policy changes
107. Infrastructure spending and policy
108. Education policy decisions
109. Student loan policy
110. Gun control positions and actions
111. Criminal justice reform efforts
112. Policing policy and reform
113. Drug policy and enforcement
114. Veteran affairs policy
115. Social Security positions
116. Medicare positions and changes
117. Medicaid policy
118. Welfare and safety net policy
119. Small business policy
120. Technology and cybersecurity policy
121. Signature healthcare legislation
122. Healthcare executive orders
123. Prescription drug pricing efforts
124. Mental health policy
125. Health insurance coverage changes
126. Public health initiatives
127. Stock market performance during term
128. National debt changes during term
129. Federal budget proposals
130. Stimulus and relief packages
131. Manufacturing and industrial policy
132. Banking and Wall Street regulation
133. Federal Reserve relations and appointments
134. Inflation during presidency
135. Corporate tax policy
136. Economic advisors and council
137. Border security positions
138. DACA and Dreamer policy
139. Refugee admission policy
140. Travel restrictions and bans
141. Deportation policy and statistics
142. Immigration executive orders
143. Family separation and detention policy
144. Legal immigration changes
145. Asylum policy
146. Immigration legislation efforts
147. Relations with China
148. Relations with Russia
149. Relations with North Korea
150. Relations with Iran
151. Relations with Israel
152. Relations with NATO allies
153. Relations with the United Kingdom
154. Relations with Mexico
155. Relations with Canada
156. Relations with Saudi Arabia
157. Middle East policy
158. Africa policy
159. Latin America policy
160. Asia Pacific strategy
161. United Nations positions
162. International agreements signed
163. International agreements withdrawn from
164. G7 summit participation
165. G20 summit participation
166. Military spending levels
167. Troop deployment decisions
168. Troop withdrawal decisions
169. Afghanistan policy
170. Iraq policy
171. Syria policy
172. Drone strike and airstrike policy
173. Special operations missions
174. Defense secretary choices
175. Nuclear weapons policy
176. Arms deals with foreign nations
177. Counterterrorism strategy
178. Supreme Court nominations
179. Supreme Court confirmation battles
180. Federal judge appointment totals
181. Judicial philosophy preferences
182. Landmark court cases during term
183. Attorney General selections
184. Department of Justice controversies
185. Executive orders signed
186. Most significant executive orders
187. Pardons and commutations granted
188. Vetoes issued
189. Emergency declarations
190. Use of executive privilege
191. Cabinet firings and resignations
192. White House staff turnover
193. Paris Climate Agreement positions
194. EPA regulation changes
195. Clean energy and fossil fuel policy
196. Oil drilling and pipeline decisions
197. Carbon emission targets
198. National park and monument decisions
199. Water and air quality regulation
200. Environmental justice initiatives
201. NAFTA and USMCA positions
202. Trans-Pacific Partnership positions
203. China tariffs and trade war
204. Steel and aluminum tariffs
205. WTO and multilateral trade positions
206. Sanctions imposed on other nations
207. COVID-19 initial response
208. COVID-19 travel restrictions
209. COVID-19 vaccine policy
210. COVID-19 economic relief
211. COVID-19 school and business closure positions
212. COVID-19 death toll and statistics during tenure
213. Impeachment proceedings
214. Special counsel investigations
215. Congressional investigations faced
216. Controversial pardons issued
217. Classified document controversies
218. Personal conduct controversies
219. Campaign finance controversies
220. Media feuds
221. Controversial public statements
222. Staff scandals and controversies
223. Relationship with House Speaker
224. Relationship with Senate leaders
225. Government shutdowns
226. Bipartisan legislation achievements
227. Key legislation signed into law
228. Debt ceiling negotiations
229. State of the Union addresses
230. Midterm election impacts
231. Secretary of State choices
232. Secretary of Treasury choices
233. Chief of Staff choices
234. National Security Advisor choices
235. Press Secretary choices
236. Cabinet diversity
237. Senior advisor appointments
238. LGBTQ rights positions
239. Racial justice positions
240. Voting rights positions
241. Abortion policy positions
242. Protest responses during presidency
243. Approval ratings over time
244. Relationship with major news networks
245. Books and memoirs written
246. Post-presidency residence and lifestyle
247. Presidential library or foundation
248. Post-presidency political involvement
249. Historians' ranking and legacy assessment
250. Signature accomplishment of presidency