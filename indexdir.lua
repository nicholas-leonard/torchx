
-- e.g. fileList = torch.indexdir("/path/to/files/", 'png', true)
-- index the directory by creating a chartensor of files paths.
-- returns an object with can be used to efficiently list files in dir
function paths.indexdir(path, extensionList, use_cache)
   extensionList = extensionList or {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   extensionList = (torch.type(extensionList) == 'string') and {extensionList} or extensionList
   
   -- repository name makes cache file unique
   local findFile = path:gsub('/', '-th-') .. '---' .. table.concat(extensionList) .. '.txt'
   findFile = paths.concat(paths.dirname(os.tmpname()), findFile)
   -- find the image path names
   local fileList = torch.CharTensor()  -- path to each image in dataset
      
   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
      
   if not (use_cache and paths.filep(findFile)) then      
      -- Options for the GNU find command
      local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
      for i=2,#extensionList do
         findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
      end

      -- run "find" on each class directory, and concatenate all
      -- those filenames into a single file containing all file paths
      local command = find .. ' "' .. path .. '" ' .. findOptions .. ' > "' .. findFile .. '"'
      
      os.execute(command)
   end

   -- load the large concatenated list of file paths to fileList 
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                  .. findFile .. "' |"
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. findFile .. "' |"
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any files in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")
   fileList:resize(length, maxPathLength):fill(0)
   local s_data = fileList:data()
   local count = 0
   for line in io.lines(findFile) do
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      count = count + 1
   end
   
   local obj = {tensor=fileList,cachefile=findFile}
   function obj:filename(i)
      return ffi.string(torch.data(self.tensor[i]))
   end
   function obj:size()
      return self.tensor:size(1)
   end

   return obj
end
