package assurance.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * @author fdse
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class AssuranceTypeBean implements Serializable{

    /**
     * index of assurance type
     */
    private  int index;

    /**
     * the assurance type name
     */
    private String name;

    /**
     * the price of this type of assurence
     */
    private double price;

}
