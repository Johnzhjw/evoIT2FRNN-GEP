%%
all_indicators1=zeros(2,4,10);
for i = 1:2
    for j = 1:4
        for k = 1:10
            all_indicators1(i,j,k) = Indi1((k-1)*3+1+i,1+j);
        end
    end
end

%%
all_indicators2=zeros(2,4,10);
for i = 1:2
    for j = 1:4
        for k = 1:10
            all_indicators2(i,j,k) = Indi2((k-1)*3+1+i,1+j);
        end
    end
end
