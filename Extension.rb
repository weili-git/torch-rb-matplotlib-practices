
module ArrayExtension
	refine Object do
		# dimension of [1, [2, 3]] is meaningless, hence it's meaningful only for matrix.
	    def dimension
	    	self.class == Array ? 1 + self[0].dimension : 0
	    end
	end

	refine Array do
		def submat(indices)
		    raise "#{indices.class} can't be coerced into Array" if !indices.is_a?(Array)
	    	dim = self.dimension
	    	raise "submat: wrong number of arguments (given #{indices.size}, expected #{dim})" if dim != indices.size
	    	if dim == 1  # Be careful, it's not simply 'self[indices[0]]'
	    		return self[indices[0]] if indices[0].is_a?(Range)
	    		return [self[indices[0]]] if indices[0].is_a?(Integer)
	    		raise "#{indices[0].class} can't be coerced into Integer or Range"
	    	end
	    	dim.times do |d|
	    		index = indices[d]
	    		# In general, 'indices.delete_at(0)' can't be the argument, because it returns the deleted item.
	    		# Also it will change the array which might interrupt the other loops!!!
	    		new_indices = indices[1..indices.size-1] 
	    		if index.is_a?(Integer)
	    			return [self[index].submat(new_indices)]
	    		elsif index.is_a?(Range)
	    			tmp = []
	    			index.each do |r|
	    				tmp = tmp.append(self[r].submat(new_indices))
	    			end
	    			return tmp
	    		else
	    			raise "#{index.class} can't be coerced into Integer or Range"
	    		end
	    	end
		end
	end
end

if __FILE__ == $0
  using ArrayExtension
  a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  p a.submat([1..2, 0..1])
end


