function nntoolbox_select(type)
if strcmp(type,'mul'),
    
elseif strcmp(type,'bin'),
    
else
    disp('Wrong type of neural network toolbox selected')
    type
    asser(false);
end

end