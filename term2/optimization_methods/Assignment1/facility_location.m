rng(1); %change the number here to try different configurations
C = 20; %number of clients
clients = rand(2,C); %client positions
F = 15; %number of facilities
facilities = rand(2,F);

capacities = ones(1, F)*4; %maximum number of clients per facility

dx = repmat(clients(1,:),F,1)-...
        repmat(facilities(1,:)',1,C);
    
dy = repmat(clients(2,:),F,1)-...
        repmat(facilities(2,:)',1,C);
    
assignment_costs = dx.*dx + dy.*dy; %the assignment cost is the distance squared
opening_costs = ones(1, F);

%solving the task with a MIP solver
cvx_solver gurobi; %MIP solver needed, will not work without an extra license
cvx_begin
    variable y(F) binary
    variable x(F,C) binary
    
    minimize sum(y(:).*opening_costs(:))+sum(x(:).*assignment_costs(:));
subject to
    max(x,[],2) <= y;
    sum(x,2) <= capacities'.*y;
    sum(x,1) == 1;
cvx_end
y = y > 0.5;
x = x > 0.5;
cvx_optval

%visualizing the solution
figure; hold on;

colors = zeros(F,3);
colors(y,:) = lines(sum(y(:)));
scatter(facilities(1,y),facilities(2,y),100,colors(y,:),'LineWidth',3); %open facilities
scatter(facilities(1,~y),facilities(2,~y),100,'k','LineWidth',1); %closed facilities
[~,assignments] = max(x,[],1);
scatter(clients(1,:)',clients(2,:)', 30,colors(assignments,:),'filled');
daspect([1 1 1]);
hold off;
saveas(gcf,'result.png');

