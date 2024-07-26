# FFRankNet
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Project Description</a>
    </li>
    <li><a href="#results">Results</a></li>
    <li><a href="#implementation">Implementation</a></li>
    <li><a href="#technology">Technology</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- Description -->
## Description

I'm an avid football fan, especially when it comes to Fantasy Football. FF is a deeply statistical endeavor, and there's countless models which are designed to predict results, whether it be season-by-season, week-by-week, and game-by-game. 

However, these models are typically focused on a point value. My philosophy has always been not to focus on the points, but the ordering of performance of players. If you consistenly have a player who is a "top" scorer, you will perform well more often, compared to those who have players with extremely high highs. 

Therefore, I'd wanted to order players for the reason in their respective position. After some research, I ran across the RankNet, which is a learning to rank framework.

<!-- Results -->
## Results

## Quarterback Ranking Prediction (Top 15):
```
Lamar Jackson
Josh Allen
Jalen Hurts
Dak Prescott
Patrick Mahomes
Jared Goff
Brock Purdy
Jordan Love
Tua Tagovailoa
C.J. Stroud
Matthew Stafford
Justin Fields
Sam Howell
Trevor Lawrence
Baker Mayfield
```

## Running Back Ranking Prediction (Top 30):
```
Derrick Henry
Breece Hall
Kenneth Gainwell
Rachaad White
Travis Etienne
Jaylen Warren
Alvin Kamara
Devin Singletary
Christian McCaffrey
Bijan Robinson
James Cook
Gus Edwards
Brian Robinson
Roschon Johnson
Chuba Hubbard
D'Andre Swift
Ezekiel Elliott
Isiah Pacheco
Najee Harris
Kenneth Walker III
Jonathan Taylor
Jahmyr Gibbs
Jerome Ford
Rhamondre Stevenson
Austin Ekeler
Chris Rodriguez Jr.
James Conner
Tyjae Spears
Tyler Allgeier
Josh Jacobs
```

## Wide Receiver Rankings (Top 30)
```
CeeDee Lamb
Tyreek Hill
Amon-Ra St. Brown
Puka Nacua
A.J. Brown
DJ Moore
Mike Evans
Davante Adams
Ja'Marr Chase
Stefon Diggs
Brandon Aiyuk
Nico Collins
Amari Cooper
Michael Pittman
Chris Olave
DeAndre Hopkins
DK Metcalf
Garrett Wilson
Keenan Allen
Calvin Ridley
Adam Thielen
DeVonta Smith
George Pickens
Chris Godwin
Terry McLaurin
Jaylen Waddle
Justin Jefferson
Jordan Addison
Tyler Lockett
Zay Flowers
```

## Tight End Rankings (Top 15)
```
Taysom Hill
George Kittle
Kyle Pitts
T.J. Hockenson
Trey McBride
Travis Kelce
David Njoku
Jonnu Smith
Mark Andrews
Evan Engram
Dalton Schultz
Dalton Kincaid
Tyler Conklin
Sam LaPorta
Darren Waller
```

## Kicker Rankings (Top 15)
```
Jason Sanders
Riley Patterson
Joey Slye
Jake Moody
Brandon Aubrey
Jason Myers
Matt Gay
Greg Zuerlein
Chris Boswell
Blake Grupe
Chase McLaughlin
Tyler Bass
Jake Elliott
Cameron Dicker
Harrison Butker
```

# Implementation

Ironically, I'd been steered away from this project for a while due to the lack of access to data. However, it seems I hadn't looked hard enough (see the dataset I used in acknowledgements). 

I processed data and calculated scores according to standard PPR format. After this, I combined the rows with their *next year's* scores. Then, I filtered each position by certain criteria (basic to eliminate outliers and bad examples). 

Trivially, I made pairs between each index. For a pair $(i, j)$, $\text{label}_{ij} = 1$ if $i > j$ and $0$ else.

Then, I implemented a model using TensorFlow and Keras. It was a three-layer network using Keras Dense layers ($64$, $16$, and $8$). Each piece of data was in a pair, each element in the pair was passed through these layers and then the final layer is a subtract layer. This is passed into a sigmoid, which gives it a value between $[0, 1]$. 

The loss function was quite simple, if a value of `1` was expected, it'd add `1 - y_pred`, if `0` was expected then, you'd add `y_pred`. The mean of this was taken. 

Then, I trained the model for $200$ epochs, which a batch size of $1/32$ with an `Adam` optimizer and early stopping (with a patience of $25$ iterations). 

After this, I ran the $2023$ dataset, and built a matrix from the the pairs of scores. Summing the columns gave a a vector with a predicted value, which I ordered and returned as my results.  

# Technology
- Pandas
- Numpy
- TensorFlow
- Keras

# Acknowledgements
- [NFL-Data by hvpkod](https://github.com/hvpkod/NFL-Data)
- [RankNet, LambdaRank TensorFlow Implementation by Louis Kit Lung Law](https://medium.com/swlh/ranknet-factorised-ranknet-lambdarank-explained-implementation-via-tensorflow-2-0-part-i-1e71d8923132)