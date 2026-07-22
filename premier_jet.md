Je consigne ici l'idée initiale que j'avais du projet, avant de coder.
Je mets aussi des observations diverses et variées

## 0) Unorganized thoughts
-I should probably write in english, Magic: The Gathering is a game mostly played in english.
-Regarding the 17lands data, I have seen many metrics other than GIH and AlSA (ATA) than may be slightly relevant. For instance, the GP (game played)% could be used to see whether a card is overplayed or underplayed, and give more context on the winrate.

## 1) Objectif

## 1.1) Main objective

-The goal is to gather implement an automatic card rating program, that can rate cards from newlmy released expansions.
-The program would be a neural network, that, for each card, reads its characteristics, and give a rate output between 0 and 10.
-This neural network would be trained thanks to metrics from cards of previous expansions. These metrics are gathered from 17lands in the Premier Draft sections, mainly GIH winrate, and possibly other metrics to reduce color bias (ATA or ALSA, IIH, color specific winrate if available). The card characteristics are gathered from scryfall.
-The network should be able to read card text in the context of MTG cards. It may be trained from another language model and tuned specifically for MTG, as card text is very codified.

## 1.2) Secondary objectives

-The rating could be given with a short description of the card, or main characteristics: removal, late-game, early-game drop, bomb, etc.
-This data could be used eventually for bot drafting. The previous characterics would then be useful to build a well balanced deck. 
-The card rating should eventually be set-dependant. Predicting a metagame (best colors, best decks, best synergies) would be the ultimate goal.
-More is to come surely



## 2) Tools

-17lands for the card metrics
-Scryfall for the card description
-Possibly other source for a more precise description of rulings, like release notes

## 3) Training

## 3.1) Network

-Training will be done on relatively short amount of cards. Each set has around 250 to 300 card, and there may be around 20 sets of available data.
-Because there is so few data, you cannot train the network as a regular LLM. On the other hand, MTG card text is extremely codified, so that learning should not need too much data.
-I suggest starting from an existing language model and tuning it for MTG. There may be better options, but playing around with language models is interesting.

## 3.2) 17lands metric to 0-10 rating

-Converting the 17lands metrics into a 0-10 rating that can be learned. GIH winrate is the main target, but there is a bias I would like to correct. I will give some intuition.
-GIH winrate is the main target
-It is possible to compare GIH-WR to the color pair winrate to correct a bias. Make sure not to overcorrect when the next point is taken into account.
-Contextualize GIH-WR with IIH: high GIHWR and high IIH mean that it looks like it is a strong card, and possibly that the deckis built around this card, it may be in a splashed color, as splashed colors make a deck less consistent but stronger if the corresponding card is drawn. High GIHWR and low IIH may mean that the card is a filler (bad or average card) in a strong archetype. Low GIHWR and high IIH could mean that the card is strong in a vacuum but in a weaker archetype. Low GIHWR and low IIH could mean the card is just terrible.
-A high play rate for a card could mean it is overplayed, and the deck it is played in are not optimized for it. The GIHWR may be lower than the actual strength of the card in this case. On the other hand, a low play rate would mean that the card is played in the appropriate builds, and the GIHWR should be indicative of its strength.
-A card that is picked too highly relative to its strength can make a deck weaker, and the GIHWR lower for this card. This could tend to happen for "uncommon signposts", i.e., two colored cards that define an archetype, and rare or mythic cards, that players tend to overpick. 
-There should be a very strong relationship between the pick order and the play-rate of a card. It would be interesting to look at the correlations between these data. The ALSA data for instance could then be ditched, as it is biased by players overpicking rares for raredrafting, while the play rate at least would give better ideas of card evaluation.
Conclusion: in ECL, with a Pearson correlation of -0.836, play rate is highly correlated to pick order. Pick order is however harder to compute, as it is not directly given by 17lands, sticking to play rate is better. -0.872 avec le ATA.

## 4) Input-Output

-Input: card characteristics. Mana cost, type, card text, P/T box.
-Out: Numerical rating, continuous between 0 and 10.
