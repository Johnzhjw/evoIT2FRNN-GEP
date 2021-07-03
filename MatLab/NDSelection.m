function F=NDSelection(pop)

    nPop=size(pop,1);

    DominatedCount=zeros(1,nPop);
    
    count=0;
    
    for i=1:nPop
        for j=i+1:nPop
            p=pop(i,:);
            q=pop(j,:);
            
            if Dominates(p,q)
                DominatedCount(j)=DominatedCount(j)+1;
            end
            
            if Dominates(q,p)
                DominatedCount(i)=DominatedCount(i)+1;
            end
        end
        
        if DominatedCount(i)==0
            count=count+1;
        end
    end
    
    F=zeros(count,size(pop,2));
    tmp=0;
    for i=1:nPop
        if DominatedCount(i)==0
            tmp=tmp+1;
            F(tmp,:)=pop(i,:);
        end
    end
    
end