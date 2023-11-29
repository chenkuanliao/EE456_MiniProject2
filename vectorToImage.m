function output = vectorToImage(input)
    imageMatrix = reshape(input, [], 3);
    output = reshape(imageMatrix, [32, 32, 3]);
end