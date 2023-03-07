package inside_payment.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class OutsidePaymentInfo {
    public OutsidePaymentInfo(){
        //Default Constructor
    }

    private String orderId;
    private String price;
    private String userId;

}
