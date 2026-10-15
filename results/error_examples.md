# Qualitative error examples (proposed extractor, top-10, TEST split)

`[x]` = exact (stemmed) match with a gold keyphrase. Gold keyphrases shown as annotated.

## Worst-scoring documents

**Doc 196: On the emergence of rules in neural networks**  (F1@10 = 0.00)

- Gold: associationist neural network; learns; abstract rules; neural network; state-space; symbolic knowledge; cognitive neurosciences; associationist learning
- Absent from text: cognitive neurosciences; associationist learning
- Predicted: [ ] evolutionary psychology; [ ] factor abstract rules; [ ] weight adaptation; [ ] abstract representations; [ ] finite-state machines; [ ] hidden unit space; [ ] human linguistic theory; [ ] arbitrary input symbols; [ ] grammatical knowledge; [ ] internal representations

**Doc 338: Down up -LSB- IT projects -RSB-**  (F1@10 = 0.00)

- Gold: strategic technology projects; walgreen; ford; caterpillar; victoria's secret; morgan stanley; staples
- Absent from text: strategic technology projects; walgreen; ford; caterpillar; victoria's secret; morgan stanley; staples
- Predicted: [ ] savvy cios; [ ] gloomy gdp report; [ ] second quarter; [ ] companies; [ ] economy; [ ] projects; [ ] gdp report; [ ] gdp; [ ] cios; [ ] gloomy gdp

**Doc 371: A better ballot box ?**  (F1@10 = 0.00)

- Gold: ballot box; mark-sense balloting; automatic teller machine computer-based voting system; atm kiosk style computer-based voting systems; electronic voting; online voting
- Absent from text: automatic teller machine computer-based voting system; atm kiosk style computer-based voting systems; online voting
- Predicted: [ ] internet; [ ] election officials; [ ] united states; [ ] lever machines; [ ] automatic teller machine; [ ] atm; [ ] voting products; [ ] poorer reliability; [ ] voting online; [ ] kiosk style computer-based systems

**Doc 228: Rapid microwell polymerase chain reaction with subsequent ultrathin-layer gel electrophoresis of DNA**  (F1@10 = 0.08)

- Gold: rapid microwell polymerase chain reaction; ultrathin-layer gel electrophoresis; dna amplification; large-scale genotyping; expression profiling; rapid high-performance analysis; automated electrophoresis analysis; reagent consumption; sample loading robots; computerized real time data analysis; automated microseparation; integrated scanning lif apd detection; complexation with ethidium bromide; real time imaging
- Absent from text: rapid high-performance analysis; automated electrophoresis analysis; integrated scanning lif apd detection
- Predicted: [x] large-scale genotyping; [ ] high-performance analysis; [ ] microwell polymerase chain reaction; [ ] rapid microwell polymerase chain; [ ] dna amplification technique; [ ] scanning laser-induced fluorescence-avalanche photodiode; [ ] automated microseparation technique; [ ] high-throughput devices; [ ] subsequent ultrathin-layer gel electrophoresis; [ ] dna fragments

**Doc 325: Open courseware and shared knowledge in higher education**  (F1@10 = 0.10)

- Gold: open courseware; shared knowledge; higher education; learning management systems; college; university; web courseware management systems; commercial products; ethics; internet; open-source software
- Absent from text: internet
- Predicted: [x] higher education; [ ] united states; [ ] university campuses; [ ] pedagogy; [ ] open-source software movement; [ ] academia; [ ] web-based tools; [ ] predominant models; [ ] faculty; [ ] courseware products

## Best-scoring documents

**Doc 376: Recursive state estimation for multiple switching models with unknown transition probabilities**  (F1@10 = 0.95)

- Gold: recursive state estimation; multiple switching models; unknown transition probabilities; hybrid systems; continuous-valued target states; discrete-valued regime variable; finite state markov chain; random transition probabilities; dirichlet distributions; marginal posterior distribution; maneuvering target tracking
- Absent from text: none
- Predicted: [x] multiple switching models; [x] recursive state estimation; [x] hybrid systems; [x] dirichlet distributions; [x] marginal posterior distribution; [x] random transition probabilities; [x] discrete-valued regime variable; [x] continuous-valued target states; [x] unknown transition probabilities; [x] finite state markov chain

**Doc 1956: Dynamical transition to periodic motions of a recurrent bus induced by nonstops**  (F1@10 = 0.95)

- Gold: dynamical transition; periodic motions; recurrent bus; nonstops; circular route; recurrent time; nonlinear map; complex periodic behaviors; attractor
- Absent from text: none
- Predicted: [x] dynamical transition; [x] periodic motions; [x] recurrent bus; [x] nonlinear map; [ ] dynamical behavior; [x] recurrent time; [x] attractor; [x] circular route; [x] nonstops; [x] complex periodic behaviors
