function u = l1_split_bregman_20d(R, f, params)
	%%  R       sampling mask
	%%  f       undersampled kspace
	%%  Mu      Mu
	%%  Lambda  Lambda
	%%  gamma   Gamma
	%%  nInner  number of inner loops
	%%  nOuter  number of outer loops

    if(~exist('params'))
        % Bregman Parameters
        params.Mu      	= 0.1;
        params.Lambda	= 0.01; 
        params.Gamma   	= 0.5;
        params.Ep 		= 0.7;
        params.nOuter  	= 4;
        params.nInner  	= 8;
        params.nLoops  	= 5;
    end
    
    Mu      = params.Mu;
    Lambda  = params.Lambda;
    Gamma   = params.Gamma;
    Ep      = params.Ep;
    nOuter  = params.nOuter;
    nInner  = params.nInner;
    nLoops  = params.nLoops;
    %%
    
	%% Get size of the kspace  in dimx dimy and temporals 
	dim = size(f);

	%% Reserve memory for the auxillary variables
	f0 = f;
% 	ft = zeros(dim);
% 	u  = zeros(dim);
% 	ut = zeros(dim);
 	pu = zeros(dim);
	x  = zeros(dim);
	y  = zeros(dim);
% 	z  = zeros(dim);

	bx = zeros(dim);
	by = zeros(dim);
% 	bz = zeros(dim);
% 
% 	tx = zeros(dim);
% 	ty = zeros(dim);
% 	tz = zeros(dim);
% 
% 	dx = zeros(dim);
% 	dy = zeros(dim);
% 	dz = zeros(dim);
% 
% 	rhs  = zeros(dim);
% 	murf = zeros(dim);
    
	%% Initialize the murf
	murf = R.*f0;
	murf = ifftn(murf);
	murf = Mu*murf;
    
%     scale = dim(1);
%     murf  = scale.*murf;
	%% Initial u
	u = ifftn(f0);
    imagesc(abs(u)); colormap gray; axis square off; drawnow;
    
    factor = 0.5;
    weight = abs(u).^factor;
    weight = weight./max(weight(:));
    %%
    tStart = tic;
     phi=zeros(dim);
    for outer = 1:nOuter
       
		for inner = 1:nInner;
			fprintf('Outer loop %dth, Inner loop %dth\n', outer, inner);
			%% Update rhs
			tx = Dxt(x-bx);
			ty = Dyt(y-by);
			
			%% !!!
			rhs = murf + Lambda*tx + Lambda*ty;
			
			%% !!!  Modified (Damped) Richardson iteration for solving linear system
			for loop = 1:nLoops;  
				Ax = Mu*ifftn(R.*fftn(u))       ...
                   + Lambda*Lx(u)               ...
                   + Lambda*Ly(u);
				u = u + Ep*(rhs - Ax);        
			end      
			
			
			%% Update x, y, z
			dx = Dx(u);
			dy = Dy(u);

			[x,y] = shrink2(dx + bx, dy + by,  1/Lambda);
% 			[z]   = shrink1(dz + bz, 1/Gamma);
% 			[x,y,z] = shrink3(dx + bx, dy + by, dz + bz,  1/Lambda);
			%% Update Bregman variable
			bx = bx+dx-x;
			by = by+dy-y;

            
            
            imagesc(abs(u)); colormap gray; axis square off; drawnow;
			%% Epsilon
            du   = abs(u-pu);
            du   = scale1(du);
            diff = sum(du(:))/numel(du)
%%            
            if diff < 1e-6
                break
            end
            %% Update u
%             x = fftn(u);
%             residual = f0 - R.*x;
%             residual = residual.*R;
% 
%             gradient = ifftn(residual);
% %             gradient = ifft(gradient,[],3);
%             gradient = -weight.*gradient;     
%             %% Conjugate Gradient
%             if(inner>1)
%                 beta = gradient(:)'*gradient(:)/(gradient_prev(:)'*gradient_prev(:));
%                 direction = -gradient+beta*direction_prev;
%             else
%                 direction = -gradient;
%             end
%             direction_prev = direction;
%             gradient_prev  = gradient;
%             
%             z = weight.*direction;
% %             z = fft(z,[],3);
%             z = fftn(z);
%             z = R.*z;
%             alpha = (residual(:)'*z(:)-Lambda*(direction(:)'*phi(:))) ...
%                      /(z(:)'*z(:)+Lambda*direction(:)'*direction(:));
%             phi = phi+alpha*direction;
%             x = weight.*phi;
%             
%             u = ifftn(x);
%
%%%%
%%%%
%             wu = 1./(u+Ep); 
%             wu = u; 
%             %wu = bx + by + bz;
%             wu = x + y;
%             wu = abs(wu)+Ep;
%             wu = wu./(max(wu(:)));
%             u  = wu.*u;
%             u  = u./wu;
		end %% End inner loop
        
        weight = abs(u).^factor;    
        weight = weight/max(weight(:));      
%         u = u.*weight;
		%% Update murf
		ft = fftn(u);
		f  = f+f0-R.*ft;
%         f  = f0-R.*ft;
% 		f = f/scale;
		murf = Mu*ifftn(R.*f);
% 		murf = murf*scale;
    end %% End outer loop
	tEnd = toc(tStart);
	fprintf('%d minutes and %f seconds\n',floor(tEnd/60),rem(tEnd,60));
return;


% !!! Central differences
function d = Dx(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(:,2:cols-1,:) = (u(:,3:cols,:)-u(:,1:cols-2,:))/2.0;
	d(:,1,:) = u(:,2,:)-u(:,1,:); % No wrap-around
	d(:,cols,:) = u(:,cols,:)-u(:,cols-1,:); % No wrap-around
return

% !!!
function d = Dxt(u)
	d = -Dx(u);
return

function d = Dy(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(2:rows-1,:,:) = (u(3:rows,:,:)-u(1:rows-2,:,:))/2.0;
	d(1,:,:) = u(2,:,:)-u(1,:,:);
	d(rows,:,:) = u(rows,:,:)-u(rows-1,:,:);
return

function d = Dyt(u)
	d = -Dy(u);
return


function d = Dz(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(:,:,2:tems-1) = (u(:,:,3:tems)-u(:,:,1:tems-2))/2.0;
	d(:,:,1) = u(:,:,2)-u(:,:,1); % No wrap-around
	d(:,:,tems) = u(:,:,tems)-u(:,:,tems-1); % No wrap-around
return

% !!!
function d = Dzt(u)
d = -Dz(u);
return

% !!! Laplacian along x direction
function d = Lx(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(:,2:cols-1,:) = u(:,1:cols-2,:)-2*u(:,2:cols-1,:)+u(:,3:cols,:);
	d(:,1,:) = u(:,2,:)-u(:,1,:);
	d(:,cols,:) = u(:,cols-1,:)-u(:,cols,:);
	d = -1*d;
return

% !!! Laplacian along y direction
function d = Ly(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(2:rows-1,:,:) = u(1:rows-2,:,:)-2*u(2:rows-1,:,:)+u(3:rows,:,:);
	d(1,:,:) = u(2,:,:)-u(1,:,:);
	d(rows,:,:) = u(rows-1,:,:)-u(rows,:,:);
	d = -1*d;
return

% !!! Laplacian along z direction
function d = Lz(u)
	[rows,cols,tems] = size(u);
	d = zeros(rows,cols,tems);
	d(:,:,2:tems-1) = u(:,:,1:tems-2)-2*u(:,:,2:tems-1)+u(:,:,3:tems);
	d(:,:,1) = u(:,:,2)-u(:,:,1);
	d(:,:,tems) = u(:,:,tems-1)-u(:,:,tems);
	d = -1*d;
return



function [xs,ys,zs] = shrink3(x,y,z,lambda)
	s = sqrt(x.*conj(x)+y.*conj(y)+z.*conj(z));
	ss = s-lambda;
	ss = ss.*(ss>0);

	% s = s+(s<lambda);
	ss = ss./s;

	xs = ss.*x;
	ys = ss.*y;
	zs = ss.*z;
return;
%%
function [xs,ys] = shrink2(x,y,lambda)
	s = sqrt(x.*conj(x)+y.*conj(y));
	ss = s-lambda;
	ss = ss.*(ss>0);

	% s = s+(s<lambda);
	ss = ss./s;

	xs = ss.*x;
	ys = ss.*y;
return;

function [xs] = shrink1(x,lambda)
	s = sqrt(x.*conj(x));
	ss = s-lambda;
	ss = ss.*(ss>0);

	% s = s+(s<lambda);
	ss = ss./s;

	xs = ss.*x;
return;

function y =  print(x);
	[rows,cols,tems] = size(x);
	fprintf('===========================================================');
	x(1,1,1)
	x(2,1,1)
	x(3,1,1)
	x(4,1,1)
	% x(rows,cols,tems)
	% fprintf('%f \n', x(1,1,1));
	% fprintf('%f \n', x(2,1,1));
	% fprintf('%f \n', x(3,1,1));
	% fprintf('%f \n', x(rows, cols, tems));
	% fprintf('%e \t %e\n', real(x(rows, cols, tems)), imag(x(rows, cols, tems)));
	% fprintf('%e \t %e\n', real(x(1, 1, 1)), imag(x(1, 1, 1)));
	% fprintf('%e \t %e\n', real(x(2, 1, 1)), imag(x(2, 1, 1)));
	% fprintf('%e \t %e\n', real(x(3, 1, 1)), imag(x(3, 1, 1)));
	% fprintf('%e \t %e\n', real(x(4, 1, 1)), imag(x(4, 1, 1)));
	% fprintf('%e \t %e\n', real(x(rows, cols, tems)), imag(x(rows, cols, tems)));
return;
