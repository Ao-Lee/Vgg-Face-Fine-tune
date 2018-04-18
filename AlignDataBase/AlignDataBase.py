from align import AlignDatabase, GetAlignFuncByBoundingBox

if __name__ == '__main__':
    
    source = 'E:\\DM\\Faces_origin\\Data\\PCD\\1'
    target = 'E:\\DM\\Faces_origin\\Data\\PCD\\2'
    
    
    F = GetAlignFuncByBoundingBox(output_size=224, margin=24)
    AlignDatabase(source, target, align_func=F)
    
