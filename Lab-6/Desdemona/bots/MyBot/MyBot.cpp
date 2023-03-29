/*
 * @file botTemplate.cpp
 * @author Arun Tejasvi Chaganty <arunchaganty@gmail.com>
 * @date 2010-02-04
 * Template for users to create their own bots
 */
#define INT_MIN -2147483648
#define INT_MAX +2147483647
#include <chrono>
#include<vector>
#include "Othello.h"
#include "OthelloBoard.h"
#include "OthelloPlayer.h"
#include <cstdlib>
using namespace std;
using namespace Desdemona;

auto start = chrono::steady_clock::now();

class MyBot : public OthelloPlayer
{
public:
    /**
     * Initialisation routines here
     * This could do anything from open up a cache of "best moves" to
     * spawning a background processing thread.
     */
    MyBot(Turn turn);

    /**
     * Play something
     */
    virtual Move play(const OthelloBoard &board);
    virtual int minMaxAlgo(OthelloBoard &gameBoard, Turn turn,int depth,Move lastMove,int palyerNumber,int alpha,int beta);
    virtual float heuristicFunction(OthelloBoard &gameBoard);
    virtual float staticHeur(OthelloBoard &gameBoard);

private:
};

MyBot::MyBot(Turn turn)
    : OthelloPlayer(turn)
{
}

Move MyBot::play(const OthelloBoard &board)
{
    start = chrono::steady_clock::now();
    list<Move> possibleMoves = board.getValidMoves(turn);
    Move currentBestMove = *possibleMoves.begin();
    int bestScore = INT_MIN;
    int searchDepth = 5;
    OthelloBoard newBoard = OthelloBoard(board);

    for(Move currentMove : possibleMoves){
        int currHeurVal = minMaxAlgo(newBoard,this->turn,searchDepth,currentMove,0,INT_MIN,INT_MAX);
        int maxHeurVal = max(currHeurVal,bestScore);
        if(maxHeurVal==currHeurVal){
            currentBestMove=currentMove;
            bestScore=maxHeurVal;
        }else{
            currentBestMove=currentBestMove;
            bestScore=maxHeurVal;
        }
        if(currHeurVal==INT_MIN && board.validateMove(turn,currentBestMove)){
            return currentBestMove;
        }
        cout<<"\nmaxHeurVal " << maxHeurVal << endl;
    }
    newBoard.print();
    return currentBestMove;
}

int MyBot::minMaxAlgo(OthelloBoard &gameBoard, Turn lastTurn,int depth,Move lastMove,int palyerNumber,int alpha,int beta){
    //time constraint
    if (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() > 1900)
        return INT_MIN;

    //creating a new board so that the last one is not affected
    OthelloBoard newBoard = OthelloBoard(gameBoard);
    //setting the board as per the last move made by the opponent
    newBoard.makeMove(lastTurn,lastMove);
    //getting all the possible moves that we can make
    list<Move> possibleMoves = newBoard.getValidMoves(/*reversing the turn*/other(turn));

    //recursive definition of the game
    if(depth==0)//we reached the bottomMostNode
        return heuristicFunction(newBoard) /*heuristicFunction(newBoard)*/;
    
    int bestHeurVal;
    if(palyerNumber==0){
        bestHeurVal=INT_MAX;
    }else{
        bestHeurVal=INT_MIN;
    }

    //recursive minMaxAlgo

    if(palyerNumber%2!=0)
    {
//        cout << "\nHii\n";
        for(Move currentMove: possibleMoves){
            Turn nextTurn = other(turn);
            int newDepth = depth-1;
            bestHeurVal = max(bestHeurVal,minMaxAlgo(newBoard,nextTurn,newDepth,currentMove,0,alpha,beta));
            //pruning condition
            if(beta>=min(alpha,bestHeurVal))
                break;
        }
        return bestHeurVal;
    }else if(palyerNumber%2==0){
        for(Move currentMove: possibleMoves){
            Turn nextTurn = other(turn);
            int newDepth = depth-1;
            bestHeurVal = min(bestHeurVal,minMaxAlgo(newBoard,nextTurn,newDepth,currentMove,1,alpha,beta));
            if(alpha<=max(beta,bestHeurVal))
                break;
        }
        return bestHeurVal;
    }else{
        return bestHeurVal;
    }
    return 0;
}

float discCountHeur(OthelloBoard &gameBoard,int color){
    int blackDiscs = gameBoard.getBlackCount();
    int redDiscs = gameBoard.getRedCount();
    int balance = color==0 ? blackDiscs-redDiscs : redDiscs-blackDiscs;
    return balance/100;
}

float legalMoveHeur(OthelloBoard &gameBoard,Turn turn){
    int currMoves = gameBoard.getValidMoves(turn).size();
    int oppMoves = gameBoard.getValidMoves(other(turn)).size();
    int balance = currMoves-oppMoves;
    return balance;
}

float cornerHeur(OthelloBoard &board,Turn turn){
    int currCorners = 0;
    int oppCorners = 0;
    int x[]={0,0,7,7};
    int y[]={0,7,0,7};
    for(int i=0;i<4;i++){
        if(board.get(x[i],y[i])==turn){
            currCorners+=1;
        }
        if(board.get(x[i],y[i])==other(turn)){
            currCorners+=1;
        }
    }
    if(currCorners!=0 && oppCorners!=0){
        return 10*(currCorners-oppCorners);
    }
    return 0;
}

float MyBot::heuristicFunction(OthelloBoard &gameBoard){
    int heurVal=0;
    if(this->turn==BLACK){
        heurVal+=discCountHeur(gameBoard,0);
    }else{
        heurVal+=discCountHeur(gameBoard,1);
    }
    heurVal+=legalMoveHeur(gameBoard,this->turn);
    heurVal+=cornerHeur(gameBoard,this->turn);
    return heurVal;
}

float MyBot::staticHeur(OthelloBoard &gameBoard){
    vector<vector<int>> staticTable={{4,-3,2,2,2,2,-3,4},
                                {-3,-4,-1,-1,-1,-1,-4,-3},
                                {2,-1,1,0,0,1,-1,2},
                                {2,-1,0,1,1,0,-1,2},
                                {2,-1,0,1,1,0,-1,2},
                                {2,-1,1,0,0,1,-1,2},
                                {-3,-4,-1,-1,-1,-1,-4,-3},
                                {4,-3,2,2,2,2,-3,4}};
    int netHeurVal=0;
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            if(gameBoard.get(i,j)==this->turn){
                netHeurVal+=staticTable[i][j];
            }else if(gameBoard.get(i,j)==other(this->turn)){
                netHeurVal-=staticTable[i][j];
            }else{
                netHeurVal+=0;
            }
        }
    }
    return netHeurVal;
}

// The following lines are _very_ important to create a bot module for Desdemona

extern "C"
{
    OthelloPlayer *createBot(Turn turn)
    {
        return new MyBot(turn);
    }

    void destroyBot(OthelloPlayer *bot)
    {
        delete bot;
    }
}
