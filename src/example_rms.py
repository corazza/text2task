from reward_machine import RewardMachine
from rm_builder import Builder


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
        .t(0, 0, '!e&!f&!n', 0) \
        .t(0, 2, 'e&!n', 0) \
        .t(0, 3, '!e&f&!n', 0) \
        .t(2, 2, '!f&!n', 0) \
        .t(2, 4, 'f&!n', 0) \
        .t(3, 3, '!e&!n', 0) \
        .t(3, 4, 'e&!n', 0) \
        .t(4, 1, 'g&!n', 1) \
        .t(4, 4, '!g&!n', 0) \
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
        .t(0, 0, '!a&!n', 0) \
        .t(0, 1, 'a&!n', 0) \
        .t(1, 1, '!d&!n', 0) \
        .t(1, 2, 'd&!n', 0) \
        .t(2, 2, '!c&!n', 0) \
        .t(2, 3, 'c&!n', 0) \
        .t(3, 3, '!b&!n', 0) \
        .t(3, 4, 'b&!n', 1) \
        .build()


def office_e1() -> RewardMachine:
    """
    efg
    feg
    fnnneg
    (f(n)*e|ef)g
    """
    return Builder(terminal_states={1}) \
        .t(0, 0, '!e&!f&!n', 0) \
        .t(0, 2, 'e&!n', 0) \
        .t(0, 3, '!e&f&!n', 0) \
        .t(2, 2, '!f&!n', 0) \
        .t(2, 4, 'f&!n', 0) \
        .t(3, 3, '!e', 0) \
        .t(3, 4, 'e&!n', 0) \
        .t(4, 1, 'g&!n', 1) \
        .t(4, 4, '!g&!n', 0) \
        .build()
