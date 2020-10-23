function m = thresholding(m,minval,maxval)
 
    if numel(minval)==1
        m(m < minval) = minval;
    else
        m(m < minval) = minval(m < minval);
    end
    
    if numel(maxval)==1
        m(m > maxval) = maxval;
    else
        m(m > maxval) = maxval(m > maxval);
    end

% while min(m) < minval
%      m(m<minval) = minval + (minval - m(m<minval));
%      
% end
% 
% while max(m) > maxval
%      m(m>maxval) = maxval - (m(m>maxval) - maxval);
% end

end