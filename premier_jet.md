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
-

## 4) Input-Output

-Input: card characteristics. Mana cost, type, card text, P/T box.
-Out: Numerical rating, continuous between 0 and 10.
