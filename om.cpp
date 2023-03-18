/*
* @file botTemplate.cpp
* @author Arun Tejasvi Chaganty <arunchaganty@gmail.com>
* @date 2010-02-04
* Template for users to create their own bots
*/

#include <chrono>
#include "Othello.h"
#include "OthelloPlayer.h"
#include "OthelloBoard.h"


using namespace std;
using namespace Desdemona;


auto start = chrono::steady_clock::now();

class MyBot : public OthelloPlayer
{
public:
    MyBot(Turn turn);

    virtual Move play(const OthelloBoard &board);
    virtual int minMax(OthelloBoard &board, Turn turn, int depth, Move move,int choice, int beta, int alpha);
    virtual int heuristic_function(OthelloBoard &board);
};

MyBot::MyBot(Turn turn) : OthelloPlayer(turn) {}

Move MyBot::play(const OthelloBoard &board)
{
    start = chrono::steady_clock::now();
    list<Move> nextMoves = board.getValidMoves(turn);
    Move bestMove = *nextMoves.begin();
    int bestScore = -999999999;
    int depth = 3;

    for (Move nextMove : nextMoves)
    {
        OthelloBoard gameBoard = OthelloBoard(board);
        int heuristicValue = minMax(gameBoard, this->turn, depth, nextMove,0, -999999999, 999999999);

        if (heuristicValue > bestScore)
        {
            bestMove = nextMove;
            bestScore = heuristicValue;
        }

        if (heuristicValue == -999999999) // Time up, return best move so far!
            return bestMove;
    }
    printf("Selecting move : (%d, %d)\n", bestMove.x, bestMove.y);
	OthelloBoard board2 = board;
	board2.makeMove(turn, bestMove.x, bestMove.y);
	board2.print( turn );
    return bestMove;
}

int MyBot::minMax(OthelloBoard &board, Turn turn, int depth, Move move,int choice, int beta, int alpha)
{
    if (chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() > 1600)
        return -999999999;

    OthelloBoard gameBoard = OthelloBoard(board);
    gameBoard.makeMove(turn, move);
    list<Move> moveTree = gameBoard.getValidMoves(other(turn));

    if (depth == 0)
        return heuristic_function(gameBoard);

        int bestValue = (choice == 0) ? 999999999 : -999999999;

    if (choice == 0)
    {
        for (Move move : moveTree)
        {
            bestValue = min(bestValue, minMax(gameBoard, other(turn), depth - 1, move,1, beta, alpha));
            if (beta >= min(alpha, bestValue))
                break;
        }
    }
    else
    {
        for (Move move : moveTree)
        {
            bestValue = max(bestValue, minMax(gameBoard, other(turn), depth - 1, move,0, beta, alpha));
            if (alpha <= max(bestValue, beta))
                break;
        }
    }
    return bestValue;
}

int MyBot::heuristic_function(OthelloBoard &board)
{
    int oppoPossibleMoves = board.getValidMoves(other(this->turn)).size();
    int myPossibleMoves = board.getValidMoves(this->turn).size();

    int diffenceMoves = myPossibleMoves - oppoPossibleMoves;

    float a1,a2,a3,a4;
    a1=10.0*diffenceMoves;

    // if( (myPossibleMoves + oppoPossibleMoves) !=0)
    //     a2 = 100.0*(myPossibleMoves - oppoPossibleMoves)/(myPossibleMoves + oppoPossibleMoves);
    // else
    //     a2 = 0.0;

        int mycorners = 0;
        int oppcorners = 0;

        int x[] = {0, 0, 7, 7};
        int y[] = {0, 7, 0, 7};

        //calculating the number of coins on the corners
        for (int i = 0; i < 4; ++i)
        {
            if (board.get(x[i], y[i]) == this->turn)
                mycorners += 1;
            else if (board.get(x[i], y[i]) == other(this->turn))
                oppcorners += 1;
        }

        if((mycorners + oppcorners) !=0)
            a3=  100*(mycorners - oppcorners)/(mycorners + oppcorners);
        else 
            a3 = 0;

        if (this->turn == BLACK)
            a4 = 10* (board.getBlackCount() - board.getRedCount());
        else
            a4=10 * (board.getRedCount() - board.getBlackCount());

        // cout<<a1<<" "<<a2<<" "<<a3<<" "<<a4<<" "<<0.8*a1 + 0.2*a2 + 1*a3 + 0.4*a4<<endl;
        return (0.8*a1 + 1*a3 + 0.4*a4);
    
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