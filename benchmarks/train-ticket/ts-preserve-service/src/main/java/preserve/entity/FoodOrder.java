package preserve.entity;

import lombok.Data;

import java.util.UUID;

/**
 * @author fdse
 */
@Data
public class FoodOrder {

    private UUID id;

    private UUID orderId;

    /**
     * 1:train food;2:food store
     */
    private int foodType;

    private String stationName;

    private String storeName;

    private String foodName;

    private double price;

    public FoodOrder(){
        //Default Constructor
    }

}
