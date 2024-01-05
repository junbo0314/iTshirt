
function output = path_evaluate(path_sequence)
gamma = 0.5;
score = 0;
jackpot_reward_size = 160;
small_reward_size = 20;

score = score + jackpot_reward_size*gamma^(length(path_sequence) + 1);
small_reward = is_small_rewarded(path_sequence);

if ~ isempty(small_reward.index)
    for i  = 1:length(small_reward.index)
        score = score + small_reward_size*gamma^(small_reward.index(i));



    end
end

output = score;

end