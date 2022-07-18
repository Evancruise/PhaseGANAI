function [sum_xy1, sum_xy2, sum_xy12] = ComputeCovariance(sum_xy1, sum_xy2, sum_xy12,element1,element2,No_realizations)
    for i=1:No_realizations
        disp(sprintf('%d-th iteration',i)); 
        sum_xy1 = sum_xy1 + element1(:,i)*element1(:,i)';
        sum_xy2 = sum_xy2 + element2(:,i)*element2(:,i)';
        sum_xy12 = sum_xy12 + element1(:,i)*element2(:,i)';
    end
end