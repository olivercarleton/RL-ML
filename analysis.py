# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))

def question2():
    answerDiscount = 0.9 #discount factor default,
    answerNoise = 0.0 #set noise to 0 all actions deterministic
    return answerDiscount, answerNoise #return parameter pair, agent now willing to cross bridge

#(discount,noise,living reward)
def question3a():
    return (0.5, 0, -1) #short sighted agent immediate reward, doesnt fear falling

def question3b():
    return (0.3, 0.2, -1) #agent prefers +1, avoid clift risk from noise

def question3c():
    return (0.99, 0, -1) #long sighted deterministic, willing to risk for large reward

def question3d():
    return (0.9, 0.2, 0.0) #noise adds risk, safe top path for +10

def question3e():
    return (0.99, 0, 1) #living reward better than any exit, stay alive
