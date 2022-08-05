# CarRacing
Implementation of car Racing from GYM environment, using Reinforcement Learning algorithm and OpenCV package.
Firstly, three preprocessing procedures apply to images:
- Find the desired area
- Convert into GrayScale
- Convert images into binary format

Each input image is converted into: 
1. GameBoadr 
2. State Fetch 
3. Car Position

There is 2 version of this code.
1. naive algorithm: Includes 4 simple possible actions.
2. improve states: Witch contains 7 more simple possible actions. 

The following images illustrate some samples of car states during run time.
<img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/1.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/2.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/3.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/4.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/5.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/6.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/7.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/8.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/9.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/10.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/11.png" width=250><img src="https://github.com/mahsaghn/RL_CarRacing_GYM/blob/master/RunTime/12.png" width=250>
