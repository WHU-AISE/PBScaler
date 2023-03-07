package consign.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class GetPriceDomain {
    private double weight;
    private boolean isWithinRegion;

    public GetPriceDomain(){
        //Default Constructor
    }

}
