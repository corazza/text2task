from builder import Builder
from reward_machine import RewardMachine


def office_t3() -> RewardMachine:
    # 0 # initial state
    # [1] # terminal state
    # (0,0,'!e&!f&!n',ConstantRewardFunction(0))
    # (0,2,'e&!n',ConstantRewardFunction(0))
    # (0,3,'!e&f&!n',ConstantRewardFunction(0))
    # (2,2,'!f&!n',ConstantRewardFunction(0))
    # (2,4,'f&!n',ConstantRewardFunction(0))
    # (3,3,'!e&!n',ConstantRewardFunction(0))
    # (3,4,'e&!n',ConstantRewardFunction(0))
    # (4,1,'g&!n',ConstantRewardFunction(1))
    # (4,4,'!g&!n',ConstantRewardFunction(0))
    return Builder(terminal_states={1}) \
        .t(0, 0, '!e&!f', 0) \
        .t(0, 2, 'e', 0) \
        .t(0, 3, '!e&f', 0) \
        .t(2, 2, '!f', 0.0) \
        .t(2, 4, 'f', 0) \
        .t(3, 3, '!e', 0) \
        .t(3, 4, 'e', 0) \
        .t(4, 1, 'g', 1) \
        .t(4, 4, '!g', 0) \
        .build()


def office_t4() -> RewardMachine:
    # 0 # initial state
    # [4] # terminal state
    # (0,0,'!a&!n',ConstantRewardFunction(0))
    # (0,1,'a&!n',ConstantRewardFunction(0))
    # (1,1,'!b&!n',ConstantRewardFunction(0))
    # (1,2,'b&!n',ConstantRewardFunction(0))
    # (2,2,'!c&!n',ConstantRewardFunction(0))
    # (2,3,'c&!n',ConstantRewardFunction(0))
    # (3,3,'!d&!n',ConstantRewardFunction(0))
    # (3,4,'d&!n',ConstantRewardFunction(1))
    return Builder(terminal_states={4}) \
        .t(0, 0, '!a', 0) \
        .t(0, 1, 'a', 0) \
        .t(1, 1, '!d', 0.0) \
        .t(1, 2, 'd', 0) \
        .t(2, 2, '!c', 0) \
        .t(2, 3, 'c', 0) \
        .t(3, 3, '!b', 0) \
        .t(3, 4, 'b', 1) \
        .build()
