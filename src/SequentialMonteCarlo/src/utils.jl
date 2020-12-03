"""
An iterator that can be used to specify sequences from 1 to n which omit a
single value.
For example, OneToNWithout(5, 3) defines an iterator that traverses
the values 1, 2, 4, 5.

from: https://github.com/skarppinen/cpf-diff-init.git
"""
struct OneToNWithout
  n::Int
  not::Int
  length::Int
  function OneToNWithout(n::Int, not::Int)
    @assert n > 0 "`n` must be > 0";
    @assert not >= 0 "`not` must be >= 0";
    @assert not <= n "`not` must be <= `n`";
    new(n, not, not > 0 ? n-1 : n);
  end
end

function Base.iterate(iter::OneToNWithout, state::Tuple{Int, Int} = (1, 0))
  element, count = state;
  count >= iter.length && return nothing;
  element != iter.not && return (element, (element + 1, count + 1));
  (element + 1, (element + 2, count + 1));
end

Base.length(iter::OneToNWithout) = iter.length;
Base.eltype(::Type{OneToNWithout}) = Int;