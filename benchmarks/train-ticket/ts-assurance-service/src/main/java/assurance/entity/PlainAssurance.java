package assurance.entity;


import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;
import java.util.UUID;

/**
 * @author fdse
 */
@Data
@AllArgsConstructor
public class PlainAssurance implements Serializable {

    private UUID id;

    private UUID orderId;

    private  int typeIndex;

    private String typeName;

    private double typePrice;

    public PlainAssurance(){
        //Default Constructor
    }

}
