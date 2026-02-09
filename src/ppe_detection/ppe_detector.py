# PPE Detection Module
# Identifies PPE items and violations

def detect_ppe(person_image, requirements):
    
    pass  # TODO: Implement PPE detection logic
"""
questions: how is the input stored/encoded?
how is information of the required PPE given? 
    is this assigned to a camera?
how can you flexibly or reliably determine what is the appropriate way a 
human and an object SHOULD intersect?

PSEUDOCODE: 
    given REQUIRED PPE
        scan image for REQUIRED PPE
        if notFound
            let notFound mean list of PPE object boxes return NOTHING 
                AND list of human boxes!=NOTHING
            flag violation
        if found
            where do the boxes intersect?            
            1. MUST have some sort of overlap OR PPE box is INSIDE human box
            2. hard(which parts of the body does the intersection must happen?)
                if the model CAN differentiate body parts, that's the hard math done for me
                otherwise, I can't imagine how to consistently judge where 2 boxes should be at relative to each other

            complicated by possibility of more than one person in view
                even worse if the image was taken in such a way that makes it look like something is worn by both
                people
"""

