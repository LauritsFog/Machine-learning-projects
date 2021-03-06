function mtxwrite(filename, A)

%MTXWRITE Write a matrix in MatrixMarket format.
%
%   MTXWRITE(FILENAME, A) writes the matrix A to the file FILENAME in
%   MatrixMarket format.
%
%SDDPACK: Software for the Semidiscrete Decomposition.
%Copyright (c) 1999 Tamara G. Kolda and Dianne P. O'Leary. 

% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the Free 
% Software Foundation; either version 2 of the License, or (at your option)
% any later version.  
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
% or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
% for more details.  
%
% You should have received a copy of the GNU General Public License along
% with this program; if not, write to the Free Software Foundation, Inc., 59
% Temple Place - Suite 330, Boston, MA 02111-1307, USA.  

[m, n] = size(A);
nnza = nnz(A);

fid = fopen(filename, 'wt');

fprintf(fid, '%%%% MatrixMarket matrix coordinate real general\n');
fprintf(fid, '%% File generated by mtxwrite from SDDPACK.\n');
fprintf(fid, '%d %d %d\n', m, n, nnz(A));

[I, J, V] = find(A);

for k = 1 : nnza
  fprintf(fid, '%4d %4d %15.6e\n', I(k), J(k), V(k));
end

fclose(fid);

return



