% Helper function for pretty printing
function print_apriori_rules(Rules,labels)
for i=1:length(Rules{1})
    if nargin < 2
        s1 = arrayfun(@(s)num2str(s), Rules{1}{i},'UniformOutput',false);
        s2 = arrayfun(@(s)num2str(s), Rules{2}{i},'UniformOutput',false);
    else
        s1 = labels(Rules{1}{i});
        s2 = labels(Rules{2}{i});
    end
    fprintf('{%s} -> {%s}\n', strjoin(s1, ', '), strjoin(s2, ', '))
end
end
