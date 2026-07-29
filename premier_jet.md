Je consigne ici l'idée initiale que j'avais du projet, avant de coder.
Je mets aussi des observations diverses et variées

## 0) Unorganized thoughts
-I should probably write in english, Magic: The Gathering is a game mostly played in english.
-Regarding the 17lands data, I have seen many metrics other than GIH and AlSA (ATA) than may be slightly relevant. For instance, the GP (game played)% could be used to see whether a card is overplayed or underplayed, and give more context on the winrate.
-Big remark: no one cares about card strength in a vacuum. The context of other cards in the set is important to judge the strentgh of a single card. Strong cards in the same colors or well supported archetypes should have boosted ratings. Find a way to remember context? Make a first run to estimate cards in a vacuum, and a second one to try judge synergies, and shared strength?
-GP WR vs GIH WR?
-IIH on its own is easier to learn as it removes the dependency on the set. However it may not be that useful to rate the actual strength of multicolored cards or buildaround cards.
-In order to properly train with the goal to evaluate a whole set, the whole set GIH may be the target...but we would have very very little data in this case.
-In MSH, B is so bad that cards somehow have a decent IIH because they have a terrible GP

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
-It is possible to compare GIH-WR to the color pair winrate to correct a bias. Make sure not to overcorrect when the next point is taken into account. However it looks like we should be careful with it, not overuse it. I would like to prioritize IIH.
-Contextualize GIH-WR with IIH: high GIHWR and high IIH mean that it looks like it is a strong card, and possibly that the deckis built around this card, it may be in a splashed color, as splashed colors make a deck less consistent but stronger if the corresponding card is drawn. High GIHWR and low IIH may mean that the card is a filler (bad or average card) in a strong archetype. Low GIHWR and high IIH could mean that the card is strong in a vacuum but in a weaker archetype. Low GIHWR and low IIH could mean the card is just terrible.
-This being said, we cannot simply compare IIH. A good card that forces you to play a bad archetype...is a not that good.
-A high play rate for a card could mean it is overplayed, and the deck it is played in are not optimized for it. The GIHWR may be lower than the actual strength of the card in this case. On the other hand, a low play rate would mean that the card is played in the appropriate builds, and the GIHWR should be indicative of its strength.
-A card that is picked too highly relative to its strength can make a deck weaker, and the GIHWR lower for this card. This could tend to happen for "uncommon signposts", i.e., two colored cards that define an archetype, and rare or mythic cards, that players tend to overpick. 
-There should be a very strong relationship between the pick order and the play-rate of a card. It would be interesting to look at the correlations between these data. The ALSA data for instance could then be ditched, as it is biased by players overpicking rares for raredrafting, while the play rate at least would give better ideas of card evaluation.
Conclusion: in ECL, with a Pearson correlation of -0.836, play rate is highly correlated to pick order. Pick order is however harder to compute, as it comes from the draft_data file, heavier and not that necessary. Sticking to play rate is better. -0.872 avec le ATA.
-Low and high ratings are given with respect to some numbers of standard deviations away from the mean.

## 3.3) The actual formula

-Work with a fixed set (ECL here at the beginning for instance). 
-First, fetch the average winrate overall, WR0. WR0 should be 5/10.
-The IIH correction is quite arbitrary, it should likely be linear. Say, GIH + \alpha * IIH, with \alpha positive. Start with \alpha = 1. The higher it is, the heavier IIH is related to GIH.
-GIH = GP + (1-p) * IIH, where p is the rate at which a given card is seen during a game (expect around 40%). This lets us fall back on the previous formula and intuition. We can actually use the score GP + \alpha_i * IIH, where \alpha_i may depend on the card. If it is 0, we have GP and the deck bias is huge. With \alpha_i = 1-p, we fall back on GIH and find some balance between the deck bias and card bias induced from IIH. As a starter, we could therefore either use GIH, GP+IIH, or GIH+IIH.
-The winrate correction with playrate sounds too arbitrary and should be left aside for now.
-I found the magic flea website which looks like a nice source to study on.

## 4) Input-Output

-Input: card characteristics. Mana cost, type, card text, P/T box.
-Out: Numerical rating, continuous between 0 and 10.

## 5) Output change

-Remind that we settled for IIH as a target, so that we could have a rating that would not depend on the set context, as much as possible.
-However our final goal is still to rate the cards depending on the context of their set.
-A possibility could be to change the output from "IIH" to the couple "(IIH, GP WR)". GP WR or GIH WR should be about the same, as their difference is somewhat proportional to IIH. This could make IIH harder to learn. The network may need extra help (attention?) to work even better, use the "set code" feature.
-Plus de couches dans le MLP ? > pas très concluant pour le moment
-Plus d'epochs de pretraining ? > Pas avec lora64
-Un mécanisme d'attention pour mieux intégrer le contexte ?
-Apprendre GP ? > sortie couple, pas vraiment de gain ni de perte prouvée sur Lora32 pretrain. Good with Lora64, mandatory even.
-Lora pretrain 64? > works well to predict the couple IIH GP, not IIH on its own. We may see this as an auxiliary task. Add more auxiliary tasks? > Play rate doesn't bring much.
-Open question: compute the mean IIH of commons and uncommons, or a similar metric, for a set, compare it to the color winrate, its correlation > in MSH, W IIH is not that good, while B is pretty good. My interpretation is that black decks are generally rather bad...
-Display: display the ratings result as a list by color, that can be opened in html, strong cards to weak cards.
-It may be better to try to learn the GP of a card relative to the color GP. IIH is really, really biased otherwise.

## 6) Attention?

-Prompt?: contexte : mon objectif est de calculer des ratings pour des cartes magic the gathering en entrainant un réseau de neurones, qui prendrait en données les cartes à rate, et renverrait ce rating inspiré des GP winrate de 17lands par exemple. L'estimation de ce GP est assez biaisée par la force des couleurs dans les sets de carte, j'aimerais donc que le réseau soit capable de lui même de trouver dans un set quelle couleur est la plus forte. Il doit donc être capable d'étudier plusieurs cartes simultanément. Actuellement, je pars d'un miniLM pretrain avec LoRA sur 64 ranks sur l'encyclopédie de scryfall. Pour l'entrainement en question, je fais du Lora 16 sur cette partie du réseau, et je rajoute une tête MLP par dessus. Pour prendre en compte le contexte, il y a une étape de plus où chaque set a ses cartes moyennées en une vecteur qui passe en donnée supplémentaire pour chaque carte du set, pour donner du contexte. J'aimerais une autre façon de faire qui permette de mieux lire les synergies entre les cartes
- La réponse au prompt précédent recommande de faire du Set-Transformer (option A) ou du GNN (option B). L'option A semble la plus simple à implémenter mais scale en N carré pour le temps/mémoire, où N est le nombre de cartes passées en même temps. Il est donc peut-être plus raisonnable de passer les cartes d'une couleur seulement, voire même de retirer les rares et mythiques si on teste ça. L'option B nécessite de construire le graphe soi-même, ce qui se rapproche en fait de l'option de simplification envisagée de l'option A, et de restreindre soi même drastiquement le nombre d'arêtes, qu idétermine le scaling aussi. Peut-être par archétype aussi.